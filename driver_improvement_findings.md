# Driver improvement findings: open-gpu-kernel-modules on DGX Spark (GB10)

Date: 2026-09-05. Tree: QuixiAI/open-gpu-kernel-modules at `7493f4cd` (610.57.04)
plus the commits described below (pool trim, UVM ATS populate, fork fixes).
Box: DGX Spark, GB10, aarch64, 20 cores, 121.7 GiB LPDDR5X unified memory,
kernel 6.17.0-1032-nvidia, installed driver 610.43.02 (Ubuntu packages, Canonical
signed), Secure Boot on, GSP firmware on, desktop session (gdm/Xorg) running.

Everything below was measured on this box. Numbers from the desktop-up state are
relative comparisons within one run; absolute figures drift a few percent with
desktop activity. Raw scripts, patches and a signed 610.43.02 build of the patched
modules are staged in `/home/eric/spark-driver-ab/`.

## 1. Summary

| # | Finding | Kind | Impact | Status |
|---|---|---|---|---|
| A | GPU-initiated faults on system-allocated (ATS) memory are serviced through `migrate_vma` one 4 KiB page at a time: 0.43 GiB/s first-touch, never a huge page, 16 M random accesses/s afterwards (220x slower than the cudaMalloc pool) | nvidia-uvm performance | Anything that lets the GPU read pageable memory directly on Spark: mmap'd model files (llama.cpp default), `cudaMallocManaged`, HMM. A 60 GB model faulted in by the GPU costs ~140 s and stays on 4 KiB pages | **A/B WIN (2026-09-05, Secure Boot off):** GPU first touch 0.42 -> 1.19 GiB/s at 4 KiB, 0.44 -> 19.6 GiB/s with THP (4096 of 4096 MiB huge-page backed, was 0), random access 17 -> 207 M acc/s; reversal leg matches stock |
| B | Freed system pages stay in the driver's page pools after the owning process exits: 55 GiB of `MemAvailable` vanished after a 64 GiB process exit, invisible in `/proc/meminfo`, no process in `nvidia-smi` | nvidia.ko correctness (memory accounting) | Breaks any user-space free-memory check on unified-memory boxes (vLLM startup). Workaround today is `drop_caches` | **Acceptance PASSED (2026-09-05):** MemAvailable 119.0 -> 119.1 GiB ten seconds after a 64 GiB process exits (stock: 119.2 -> 55.0); watermark 4096 keeps 4.3 GiB; sentinel -1 restores the old behaviour; pools-disabled composes; 20-cycle churn under a holder and a memory hog with zero new NVRM warnings |
| C | Steady-state decode data path: no driver lever. Pages are already 2 MiB end to end (RM allocation, GMMU, SMMU), translation reach is not a limiter, bandwidth is at 97% of spec, clocks and the ~90 W cap are GSP firmware policy | measurement | The 2 MiB sysmem regkey and the ZERO_FB large-page patch idea are no-ops (A/B/A NOWIN). c8-c32 tok/s multipliers are above the driver | Closed |
| D | Blocking-sync cost is CPU idle-state exit latency, not the driver's interrupt path | measurement | 7% per step with LPI-2/3 enabled, 1% with them disabled | Closed, platform tuning note |
| E | P2P fork changes: eight review findings, two of which can leave a box without a working GPU (`install.sh` unload-before-`set -e`; unsigned modules on Secure Boot) | fork code quality | Multi-GPU consumer boards; inert on Spark | Findings 1-7 fixed on this branch (two commits); 8 is not an issue by the owner's rule |

## 2. What the kernel driver controls during inference, and what was measured

The kernel modules are not in the steady-state data path. Kernels launch through
user-mode doorbells, completion is polled from user space (one `ppoll` and about one
GPU interrupt per step), and memory is mapped once at allocation. The driver matters
through: how memory is mapped, how faults on unmapped memory are serviced, what it
does at allocation and free, clock/power policy it forwards to firmware, and the
interrupt path. Each was measured.

### 2.1 Memory mapping of the cudaMalloc pool (closed)

- bpftrace on `nv_alloc_pages` during a fresh 8 GiB `torch.empty`: 8526 MiB requested
  at page size 2097152, 1 MiB at 4096/65536. The CUDA driver asks the RM for HUGE pages
  explicitly, so `_memmgrPickDefaultSysmemPageSize` (only consulted for
  `PAGE_SIZE_DEFAULT`) is never on the path. `RmEnableLargePageSizeSysmemDefault` is a
  no-op here; the previous session's A/B/A measured exactly that (NOWIN, all deltas
  within 1%).
- The SMMU is in translated mode (`iommu.passthrough=0`, GPU alone in IOMMU group 20,
  type DMA, PCIe ATS Enable+, PASID Enable+, `nvidia-smi` Addressing Mode: ATS).
  `iommu:map` tracepoint during a 4 GiB allocation: 2215 map calls of exactly 2 MiB,
  all 2M-aligned in IOVA and physical. SMMU block mappings are 2 MiB too.
- Translation reach: a DRAM-bound random gather over 4M distinct 128 B lines runs at
  3111/2964/2834/2844/3065/3033 M acc/s for spans of 1/2/4/8/16/64 GiB (512 to 32768
  two-megabyte pages). Flat. The paged-KV shape (64K random 16 KiB blocks) is
  116.8/118.1/114.9 GB/s at 1/8/64 GiB. Flat.
- The earlier "6.25x TLB penalty" was a harness artifact: its 2 MiB-region probe touched
  32K distinct lines (4 MiB, L2-resident) and its 4K-wide probe touched 4M lines
  (DRAM-bound). Holding distinct lines at 32K and widening the span 64 MiB -> 64 GiB
  costs 2x (19410 -> 9528 M acc/s); that is the only translation effect visible, and
  only on an L2-resident set.
- GPU L2 caches the sysmem pool (L2-resident random access is 6x faster than DRAM).

### 2.2 Bandwidth, clocks, power (closed)

- Read-stream 254 GB/s quiet, 97% of LPDDR5X spec. Random-sector ceiling ~3.05 G
  acc/s (~98 GB/s at 32 B sectors).
- `nvidia-smi -lgc 3003,3003` is accepted but the GPU tops out at 2580 MHz; the
  advertised 3003 is never reached under any load. Decode-shaped loads run at
  2430-2580 MHz, 20-40 W, and do not change with the lock (bandwidth/kernel bound).
- Compute-bound GEMM sits at ~2350 MHz, 82-90 W. The SW Power Capping counter grew
  9.1 s during ~20 s of GEMM although the per-sample flag never read Active: the cap
  is real and enforced by GSP. No control exists: power limit N/A, no hwmon, no
  power_cap, no nvpmodel. `RMPowerFeature`, `RMPriorityBoost`,
  `RmBootGspRmWithBoostClocks` are unrelated.
- Batch-32 decode step is 27% slower than batch-8 while using 192 GB/s and 6 TFLOPS:
  that is the GEMM kernel's M=32 tile choice in user space, not the driver.

### 2.3 Wait path, interrupts, CPU (closed)

- Per-process profile of a launch-heavy loop: 67% of on-CPU samples in kernel poll
  syscalls, 13% libcuda, 19% vdso. Steady state: ~62 `ppoll`/s and ~1 GPU interrupt per
  step. Driver share of a 7.7 ms step is tens of microseconds.
- Tiny kernel + synchronize: 6.5 us spin, 16.9 us blocking. The 48-layer loop with one
  sync per step: spin 7.68 ms, blocking 8.22 ms (+7%); with cpuidle states 1-3 disabled:
  7.76 vs 7.82 ms (+1%). The gap is LPI-2/LPI-3 exit latency (231/433 us), a platform
  tuning note (`cpuidle.off`, or a busy-poll serving loop), not the driver.
- GPU MSI on CPU7 via threaded IRQ (`irq/343-nvidia`, prio 90); cpufreq performance
  governor at 2.8 GHz; `EnableMSI=1`; `DynamicPowerManagement=3` is moot while
  persistence mode and the display hold the device.
- Co-tenant: a second CUDA context at ~10% duty (three 2048^3 GEMMs per 10 ms) costs
  7% per step (8.41 -> 9.01 ms). Timeslicing (default TSG timeslice
  `TIMEOUT_128 << SCALE_3`) shares proportionally; no pathological overhead.

### 2.4 Allocation, zeroing, pools, load time (measured)

- 16 GiB cudaMalloc + first GPU write: cold pools 0.354 s (45 GiB/s, includes
  `__GFP_ZERO` in the kernel allocator) vs warm pools 0.099 s (161 GiB/s). A 100 GB
  model: ~2.2 s cold, ~0.6 s warm. `InitializeSystemMemoryAllocations=1`.
- `EnableSystemMemoryPools=0x211`: 4K, 64K and 2M pools; pool pages are physically
  adjacent often enough that the IOMMU core coalesces them into 8 and 16 MiB map calls.
- `echo 2 > /proc/sys/vm/drop_caches` reclaimed 100 GiB from the pools in 0.66 s: the
  shrinker works; the problem is only that nothing triggers it at process exit (B).
- Driver reload: `nvidia-nvlink` init to `nvidia-drm` loaded is ~3 s (GSP boot).

## 3. Finding A: ATS fault servicing on integrated GPUs

### 3.1 Numbers (8 GiB and 4 GiB anonymous mmap buffers, identical CUDA kernels; reference is the cudaMalloc pool in the same run)

| Path | First touch | Read stream | Random 4K-page lines | 16 KiB blocks |
|---|---|---|---|---|
| cudaMalloc pool (reference) | 0.35 s/16 GiB cold | 233 GB/s | 3515 M acc/s | 116 GB/s |
| ATS, GPU first write, THP off | 9.37 s / 4 GiB = **0.43 GiB/s** | 169 GB/s | **16 M acc/s** | 106 GB/s |
| ATS, GPU first write, `MADV_HUGEPAGE` | 9.11 s = 0.44 GiB/s, **0 MiB THP** | 168 GB/s | 16 M acc/s | 105 GB/s |
| ATS, CPU first write, THP off | 1.24 s = 3.2 GiB/s | 168 GB/s | 17 M acc/s | 107 GB/s |
| ATS, CPU first write, `MADV_HUGEPAGE` | 0.14 s = **28 GiB/s**, 4096 MiB THP | 164 GB/s | **208 M acc/s** | 105 GB/s |
| `cudaHostRegister`'d pinned | n/a | 169 GB/s | 15 M acc/s | 104 GB/s |
| Pageable H2D copy into the pool | 55 GiB/s | | | |

Observations: (1) the GPU-initiated fault path is 7x slower than a CPU touch at 4 KiB
and 65x slower than a CPU touch with THP; (2) GPU-initiated faults never produce THP
even under `MADV_HUGEPAGE`; (3) with THP the GPU's random access to ATS memory is 13x
better, still 17x behind the pool; (4) under ATS, `cudaHostRegister` does not create a
driver mapping at all (behaves exactly like unregistered memory); (5) streaming reads
through ATS run at 72% of the pool and 16 KiB blocks at 91%, so weight streaming from
mmap'd files is a moderate loss, random KV access from such memory is catastrophic.

### 3.2 Mechanism

The faults are serviced by nvidia-uvm, not by the SMMU page-request queue: a 2 GiB
GPU fault-in raised 4095 interrupts on the UVM line (~128 pages per batch, ~1.1 ms
per batch) and none on the SMMU. `uvm_ats_service_faults` ->
`uvm_ats_service_faults_region` -> `service_ats_requests` -> `uvm_migrate_pageable`.
On an integrated GPU (`is_integrated_gpu`, set for GB10 in `uvm_blackwell.c:128,137`)
`ats_batch_select_residency` picks the closest CPU NUMA node and
`residency_id = UVM_ID_CPU`: there is nothing to migrate, yet the range goes through
`migrate_vma_setup` and `alloc_pages_node(..., order 0)` per 4 KiB page
(`uvm_migrate_pageable.c:316`). `migrate_vma` cannot form a THP, and the per-page setup
is where the 9 us/page goes. The GPU TLB is then flushed on every serviced region
because `PAGE_SIZE == 4K` (`uvm_ats_faults.c:573`).

### 3.3 Patch (in the tree, `kernel-open/nvidia-uvm/uvm_ats_faults.c`)

In `service_ats_requests`, when servicing faults on an integrated GPU with CPU
residency, populate the range in place with `uvm_populate_pageable_vma()` (the same
permissions and flags the migrate path would have used) and return, leaving
access-counter servicing on the migration path. `uvm_populate_pageable_vma` uses
`handle_mm_fault()` per page, which honours THP (first fault fills the 2 MiB PMD, the
next 511 find it present) and skips `migrate_vma` entirely. Expected: first-touch from
0.43 GiB/s to the 3-28 GiB/s range depending on THP, and 13x better random access
when THP applies. Note `uvm_populate_pageable_vma` calls `handle_mm_fault` twice per
page as a work-around for a v6.6 arm64 AF-bit bug; whether 6.17 still needs the second
pass is worth checking, it doubles the populate cost.

Builds on both 610.43.02 (system source) and 610.57.04 (this tree). Same warnings as
stock (two `nv-mmap.c` `-Waddress`, the aarch64 `os_dbg_breakpoint` `#warning`).

### 3.4 A/B result (stock -> patched -> stock, desktop up, only nvidia-uvm swapped)

| | Stock | Patched | Stock again |
|---|---|---|---|
| GPU first touch, THP off | 0.42 GiB/s | **1.19 GiB/s** | 0.38 GiB/s |
| GPU first touch, `MADV_HUGEPAGE` | 0.44 GiB/s, 0 MiB THP | **19.6 GiB/s, 4096 MiB THP** | 0.38 GiB/s, 0 MiB THP |
| Random 4K-page lines after GPU touch + THP | 17 M acc/s | **207 M acc/s** | 17 M acc/s |
| CPU first touch with THP (control) | 30.0 GiB/s | 30.1 GiB/s | 28.2 GiB/s |
| cudaMalloc pool stream (control) | 256 GB/s | 256 GB/s | 255 GB/s |

Verdict: WIN. GPU-initiated faults now produce transparent huge pages exactly as a CPU
touch does, the fault-in rate is 2.8x better at 4 KiB and 45x better with THP, and the
GPU's random access to that memory afterwards is 12x better. Controls did not move.
Raw log: `/home/eric/spark-driver-ab/ab.log`. The remaining gap to the CPU touch at
4 KiB (1.19 vs 5 GiB/s) is the per-page fault-buffer round trip plus the double
`handle_mm_fault` pass; both are follow-ups, not blockers.

### 3.5 Workaround available today, no driver change

Touch the memory from the CPU first with `madvise(MADV_HUGEPAGE)` (THP is `madvise`
on this box) before handing it to the GPU. That is the 28 GiB/s / 208 M acc/s row.
For file-backed weights the equivalent is reading them into a THP-backed anonymous
buffer or, better, into the cudaMalloc pool (55 GiB/s H2D from pageable memory).

## 4. Finding B: pooled pages after process exit

Reproduced on stock 610.43.02 with the desktop up and persistenced holding the device:
`MemAvailable` 107.0 GiB before, 51.7 GiB ten seconds after a 64 GiB GPU process
exited, zero processes in `nvidia-smi`. Matches `drain-on-last-client-close.md`
section 1 and the NVIDIA forum regression report it cites.

Implemented as specified in that document, with these notes:

- `nv-vm.c`: `nv_mem_pool_reclaim()` factors the shrinker scan's move-and-free out
  (dirty first, then clean, mutex dropped before freeing, `WARN_ON` if freed would
  exceed `pages_owned`); `nv_mem_pool_shrinker_scan` now calls it;
  `nv_trim_page_pools(retain_pages)` walks nodes and orders from
  `NV_MAX_PAGE_ORDER` down, keeps entries until the running total of
  `pages_owned << order` reaches the watermark, frees the rest, logs at
  `NV_DBG_MEMINFO`. The scrubber's in-flight entry is left alone as the document
  describes.
- `nv-reg.h`: `NVreg_SystemMemoryPoolRetainMB`, default 0, `0xFFFFFFFF` disables,
  documented next to `EnableSystemMemoryPools`; `NV_DEFINE_REG_ENTRY_GLOBAL` and
  params-table entry added. `nv-proto.h`: prototype.
- `nv.c`: `nv_trim_page_pools_on_client_close()` converts MiB to order-0 pages,
  honours the sentinel, and is called (a) in `nvidia_close_callback` after
  `up(&nvl->ldata_lock)` in the non-removal branch, after the `bRemove` handling, and
  (b) at the end of `nvidia_ctl_close`. Point (b) is an addition to the document's
  section 4.3: RM clients and the allocations they own belong to the control file, so
  a process that closes its device fd before its control fd would otherwise be trimmed
  before its pages reached the pools. Both closes are sleepable and hold no driver
  lock at the call site.
- Behaviour to be aware of: with the default watermark every client close trims,
  including `nvidia-smi` polls while a serving engine is running. The engine's own
  future allocations then pay the cold path (45 vs 161 GiB/s). The watermark exists for
  operators who want both; the default keeps `MemAvailable` truthful, as the document
  asked. The document's line numbers were for `7493f4cd`; hunks applied to 610.43.02
  with offsets of -4/-13 lines only.
- Section 3b of the request document (added 17:18, after the implementation) was
  re-checked: the trim is only reachable from file-close paths, which run while the
  module is pinned, and module exit stops the deferred-close queue (nv.c:549) before
  `nv_destroy_page_pools` (nv.c:552), so it can never run during or after teardown;
  the diff leaves `nv_mem_pool_destroy` and the shrinker teardown untouched, so a
  later rebase onto upstream PR 1004 stays trivial. Nothing upstream trims or caps
  the pools; this is new work, and the document's suggestion of an upstream issue
  plus PR once validated stands.
- Acceptance results (driver `610.43.02` built from `/usr/src/nvidia-610.43.02` with the
  patch, loaded from a modprobe override directory with gdm stopped, stock restored after;
  raw logs `pooltrim.run2.log` and `pooltrim.log` in `/home/eric/spark-driver-ab/`):

  | # | Configuration | MemAvailable before | 10 s after 64 GiB process exit | Result |
  |---|---|---|---|---|
  | 1 | stock 610.43.02 | 119.2 GiB | 55.0 GiB | failing case reproduced |
  | 2 | patched, `RetainMB=0` (default) | 119.0 GiB | 119.1 GiB | PASS |
  | 3 | patched, `RetainMB=4096` | 119.2 GiB | 114.9 GiB (4.3 GiB retained) | PASS |
  | 4 | patched, `RetainMB=-1` (disable) | 119.0 GiB | 54.4 GiB (phantom returns) | PASS |
  | 5 | patched, `EnableSystemMemoryPools=0` | 118.8 GiB | 119.2 GiB, no oops | PASS |
  | 6 | churn: 20 x alloc 32 GiB/free/exit, 16 GiB holder, host memory hog | | 117.4 GiB after, 0 new NVRM warnings | PASS |
  | 7 | vLLM startup check | | not run (SlimServe out of scope this session; `RUN_SLIMSERVE_TEST=1` in the script) | skipped |

  Two things learned running it: the module parameter is a signed 32-bit int, so the
  disable sentinel must be passed as `-1` (`0xFFFFFFFF` is rejected with ERANGE at load;
  `nv-reg.h` and the README now say so), and NVreg parameters are registered with
  permission 0, so they never appear under `/sys/module/nvidia/parameters/`; read them
  from `/proc/driver/nvidia/params`. The box's udev rule reloads the packaged
  nvidia-uvm/modeset/drm the instant `nvidia.ko` binds, so a test loader must make
  `modprobe` itself resolve to the patched modules (an `updates/` override directory plus
  `depmod`) rather than race udev with `insmod`.
- Acceptance tests 2-7 originally needed the patched `nvidia.ko` loaded, which needs every GPU user
  gone (desktop down) and the MOK enrolled. `/home/eric/spark-driver-ab/run_pooltrim_ab.sh`
  runs tests 1-6 detached, restores stock and gdm, and logs to `pooltrim.log`. Test 6
  (20 allocate/free/exit cycles against a 16 GiB holder and a host memory hog, `stress-ng`
  if installed, a numpy hog otherwise, counting new NVRM warnings in dmesg) is included;
  test 7 (the vLLM startup check) runs only with `RUN_SLIMSERVE_TEST=1` since it depends
  on the SlimServe tree.

## 5. What is needed to run the A/Bs

Secure Boot was turned off on 2026-09-05 (lockdown none), so the staged modules load
without enrolment; the notes below apply if it is turned back on. Secure Boot enforces
module signatures (`lockdown: integrity`, `sig_enforce=Y`) and
the only enrolled key is Canonical's. The staged modules are signed with
`/var/lib/shim-signed/mok/MOK.der`, which is present but not enrolled
(`mokutil --test-key` says so; `insmod` answers "Key was rejected by service").
Enrolling needs the console once:

```
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der   # choose a one-time password
sudo reboot                                              # blue MokManager screen: Enroll MOK -> Continue -> password
```

Then, with the desktop up (only nvidia-uvm is swapped, in memory, stock restored at
the end):

```
/home/eric/spark-driver-ab/run_uvm_ats_ab.sh
```

and, detached, taking the desktop down and restoring it (the launching session dies
if it runs under gdm; read `pooltrim.log` afterwards):

```
/home/eric/spark-driver-ab/run_pooltrim_ab.sh
```

## 5b. Installed permanently on the Spark (2026-09-05)

`install.sh` from this tree was not used: it would install 610.57.04 modules against the
box's 610.43.02 user space, and NVIDIA requires the two to match exactly. Instead the two
Spark commits (pool trim, UVM ATS populate; the nv-reg.h ReBAR default left at stock) were
applied to a copy of the packaged source, `/usr/src/nvidia-spark-610.43.02`, with a
`dkms.conf` (package `nvidia-spark`, version 610.43.02, `AUTOINSTALL=yes`, and
`KERNELRELEASE=` appended to the make line because DKMS 3.x injects that variable and it
flips NVIDIA's Makefile into its Kbuild-include mode). `dkms install --force` placed all
five modules in `/lib/modules/6.17.0-1032-nvidia/updates/dkms/`, which depmod searches
before the packaged `kernel/nvidia-610-open/`. The initramfs does not carry the nvidia
modules on this box, so nothing there needed rebuilding. The driver was reloaded with the
desktop stopped and confirmed: `/proc/driver/nvidia/params` shows
`SystemMemoryPoolRetainMB: 0`, `nvidia_uvm` srcversion matches the A/B-tested build, and a
64 GiB process exit returns MemAvailable within seconds. `dkms status` reports
`nvidia-spark/610.43.02 ... installed`.

To revert: `sudo dkms remove -m nvidia-spark -v 610.43.02 --all` then reload the driver or
reboot; the packaged modules are untouched underneath. A kernel upgrade rebuilds the
patched modules automatically through DKMS as long as the 610.43.02 source still compiles
against the new kernel.

## 6. Smaller observations

- The GPU's PCIe config space reports `LnkSta: 2.5GT/s x1 (downgraded)` against
  `LnkCap 2.5GT/s x16`. The data path is the on-die fabric (254 GB/s cannot cross a
  Gen1 x1 link); the link fields are a stub. `pci=pcie_bus_safe` and MPS 256 B are
  irrelevant for the same reason.
- THP is `madvise` system-wide and `AnonHugePages` is 0 on an idle box. Nothing on the
  box uses huge pages unless it asks. Combined with finding A this is the single
  biggest trap for "just mmap the model" workflows on Spark.
- The `nvidia-uvm` ATS servicing flushes the whole GPU TLB for every serviced region
  when the OS page is 4 KiB (`uvm_ats_faults.c:573`); a 64 KiB-page kernel avoids the
  hammer but is a much larger platform change.
- `NVreg_EnableSystemMemoryPools` keeps a 64K pool that the CUDA driver essentially
  never uses on this box (1 MiB of 64K requests against 8.5 GiB of 2M). Harmless.
- Identity IOMMU domain for the GPU (`echo identity > /sys/kernel/iommu_groups/20/type`
  with the driver unbound, no reboot) was deliberately not tested: the measured
  translation effect is bounded to the L2-resident regime and it needs the desktop
  down again.
- `drain-on-last-client-close.md` section 7 suggests installing over the DKMS
  modules; the staged kit deliberately does not install anything. Loading from
  `/home/eric/spark-driver-ab/modules` with `insmod` leaves the packaged driver and the
  initramfs untouched, and a reboot returns to stock.

## 7. P2P fork review (from the 2026-09-05 review session, re-checked against `git diff e4a5faa2..HEAD`) and what was done

Not Spark work: on this box every P2P change is inert (single GPU, no BAR1, both HAL
arms identical) and the fork behaves as stock 610.57.04. The fixes are on this branch
because they concern the fork's own install path and tooling for multi-GPU boards.

1. `install.sh` ran `rmmod` before `set -e`, so a failed unload silently installed nothing
   and `nvidia-smi` printed the old version. **Fixed:** build first, unload second, abort if
   `nvidia` is still loaded, verify the running version equals the build.
2. Modules were installed unsigned; on a Secure Boot box the machine comes up GPU-less.
   **Fixed:** `install.sh` signs every `.ko` with the MOK when Secure Boot is on and aborts,
   before unloading anything, if there is no key or it is not enrolled. README documents the
   one-time enrolment.
3. `tools/resize-bar1.sh` removed whole root-port subtrees, killed the display server
   silently, and never checked that BAR1 was assigned. **Fixed:** it enumerates every function
   under each GPU root port and aborts (reloading the driver at the old size) if a non-NVIDIA
   device would be removed, unless `RESIZE_BAR1_FORCE=1`; after the rescan it reads each GPU's
   BAR1 from sysfs and greps dmesg for `can't assign; no space`, exiting non-zero if the
   resize did not take; the header documents the `fuser -k`.
4. `nvidia-resize-bar1.service` was ordered after the target containing the units it kills.
   **Fixed:** `DefaultDependencies=no`, after `systemd-modules-load.service`, before
   `nvidia-persistenced.service`, `display-manager.service` and `multi-user.target`, with a
   note that GPU consumers must declare `After=nvidia-resize-bar1.service` themselves.
5. `g_kern_bus_nvoc.c` had identical `if`/`else` arms at five sites. **Fixed:** each default
   arm carries a comment saying the file is hand-edited, both arms deliberately point at the
   chip-independent GH100 body, and to keep them on rebase.
6. `p2pOverride = 0x11` also disables P2P atomics. **Documented** in the README with the
   `ForceP2P=0x211` override.
7. `nv-pci.c` had the `EnableResizableBar` check commented out, leaving a dead parameter.
   **Fixed:** check restored, default flipped to 1 in `nv-reg.h`, so the opt-out works again
   and default behaviour is unchanged.
8. Commit `2bf60686` carries the original 2024 author/date on a 2026 port. Not an issue by
   the repo owner's rule: the author is aikitoria, credits are correct, the owner is only the
   committer.

## 8. Recommendations, in order

1. Enrol the MOK and run the two A/Bs. Finding A is the one with a measurable
   throughput consequence for real workloads on this box; finding B is the one that
   fixes engine restarts.
2. If A wins: also drop the second `handle_mm_fault` pass in
   `uvm_populate_pageable_vma` on kernels where the AF-bit bug is fixed, and consider
   prefetching the full 2 MiB block on first touch of a THP-eligible VMA so a single
   fault batch covers the PMD.
3. If B passes acceptance: decide the default watermark for a serving box. 0 makes
   accounting truthful; a few GiB keeps the warm-pool benefit for graph capture churn
   while `nvidia-smi` polls.
4. Fork findings 1-7 are fixed on this branch (section 7).
5. Stop looking for decode throughput in this driver on Spark. The remaining
   multipliers are KV bytes per step, the M=32 GEMM tile, and speculative decode.

## 9. Files

- Patches: the commits on this branch (see `git log`); also
  `/home/eric/spark-driver-ab/driver-patches-610.57.04.patch` and `pool-trim.patch`.
- Signed 610.43.02 modules with both patches: `/home/eric/spark-driver-ab/modules/`.
- A/B scripts: `run_uvm_ats_ab.sh`, `run_pooltrim_ab.sh`; probes: `ats_path_bench.py`
  (+ `rawkern2-build/`), `translation_reach_probe.py`, `clock_cap_probe.py`,
  `steploop.py`, `alloctime.py`.
- Earlier A/B/A and harness (page-size regkey): SlimServe
  `perf/results/2026-09-05/spark-driver-baseline/` and the notebook entry there.
