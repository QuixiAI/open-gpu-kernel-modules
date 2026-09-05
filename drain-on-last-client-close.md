# Return pooled system pages when a CUDA client exits

Request from the SlimServe DGX Spark bring-up (2026-09-05). Written for the
agent working in this tree. Everything below was verified against this tree
at `7493f4cd` (610.57.04) and against the installed driver on the Spark box
(610.43.02, kernel 6.17.0-1032-nvidia, aarch64, Secure Boot on).

## 1. The problem

On GB10 (DGX Spark) every device allocation is system memory: the GPU has no
framebuffer, `cudaMemGetInfo` total equals `MemTotal`, and weights, KV cache
and CUDA graphs all come out of the one 121.7 GiB LPDDR5X pool that the CPU,
the desktop and the OS also live in.

Since driver 590 the open kernel module keeps freed system pages in per-node,
per-page-order pools instead of returning them to Linux. After a CUDA process
exits, its pages stay in those pools until memory pressure triggers the
kernel's shrinker or an operator runs `echo 3 > /proc/sys/vm/drop_caches`.

Measured on the Spark box:

- After one vLLM boot failure plus a few smoke tests, 51 GiB of memory showed
  as used with no owning process. It was absent from every `/proc/meminfo`
  bucket (AnonPages 2.4 GiB, Cached 34 GiB, Slab 1.8 GiB) but `MemAvailable`
  was down by that amount.
- With 35 GiB reported free, a single process then allocated 96 GiB of CUDA
  memory without error, and `cudaMemGetInfo` free barely moved for the first
  40 GiB: the allocations were being satisfied from the pool. After that
  process exited the pool held about 100 GiB.
- The retention is not permanent. Several minutes later MemFree had recovered
  on its own; `drop_caches` recovers it immediately.

Consequences for serving: vLLM's startup check (`request_memory` in
`vllm/v1/worker/utils.py`) compares `psutil.virtual_memory().available`
against `gpu_memory_utilization * total`. `MemAvailable` counts page cache as
reclaimable but not these pooled pages, so the check fails with
"Free memory on device cuda:0 (66.47/121.69 GiB) on startup is less than
desired GPU memory utilization (0.85, 103.44 GiB)" after any previous CUDA
process has exited, even though the memory is there. Every engine restart on
this box hits it.

External confirmation: NVIDIA developer forum thread "Driver 590.48.01
regression: UMA memory not released after CUDA process exit (works on
580.126.09)" describes the identical symptom (about 80 GB retained, invisible
in meminfo, MemAvailable drops, nvidia-smi shows no process). NVIDIA staff
acknowledged it and said drivers past 580.126.09 are unsupported on Spark.
The DGX Spark Known Issues page and NVIDIA's own vLLM-on-Spark troubleshooting
page both give `drop_caches` as the workaround.

## 2. Where it lives in this tree

All in `kernel-open/nvidia/nv-vm.c` unless noted.

| What | Location |
| --- | --- |
| Pool struct `nv_page_pool_t` (clean_list, dirty_list, scrubber queue, `pages_owned`, mutex `lock`, shrinker) | nv-vm.c:394-409 |
| Global table `sysmem_page_pools[MAX_NUMNODES][NV_MAX_PAGE_ORDER + 1]` | nv-vm.c:411 |
| Shrinker count / scan callbacks (the only existing release path) | nv-vm.c:507-571 |
| `nv_mem_pool_move_pages`, `nv_mem_pool_free_page_list` (list helpers the scan uses) | nv-vm.c:468-503 |
| Scrubber worker: pops dirty entries under the lock, zeroes them with the lock dropped, re-adds to clean_list on the next pass | nv-vm.c:663-690 |
| `nv_mem_pool_destroy` (module unload only) | nv-vm.c:692-715 |
| `nv_mem_pool_init`: shrinker with `seeks = 1` | nv-vm.c:717-768 |
| `nv_mem_pool_free_pages`: pages go to dirty_list, `pages_owned += n`, no cap | nv-vm.c:771-843 |
| `nv_init_page_pools` / `nv_destroy_page_pools`: gated per page size by `NVreg_EnableSystemMemoryPools` | nv-vm.c:845-885 |
| `nv_free_system_pages`: routes to the pool when `at->flags.pool` and the free may sleep | nv-vm.c:903-955 |
| Regkey `NVreg_EnableSystemMemoryPools`, default 0x211 (4K, 64K, 2M pools) | kernel-open/nvidia/nv-reg.h:966-982, 1073, 1125 |
| Prototypes | kernel-open/common/inc/nv-proto.h:55-58 |
| Pool lifecycle callers (module init and exit) | kernel-open/nvidia/nv.c:552, 576, 605 |
| Client close path | kernel-open/nvidia/nv.c: `nvidia_close_callback` (about line 2148), `nv_close_device` (about 2124), `nvidia_close_deferred` (2222) |

Version check done with the tags in this tree's history: 580.65.06 has zero
references to `nv_page_pool`; 590.44.01, 595.45.04 and 610.43.02 all carry
the full implementation. So 580 had the behavior we want by not having pools.

Two facts about the existing code matter for the design:

- There is no high-water mark anywhere. `pages_owned` grows on free and only
  shrinks on reuse (`nv_mem_pool_alloc_pages`) or shrinker scan.
- The shrinker scan is already exactly the primitive we need: under the pool
  mutex it moves up to `nr_to_scan` entries from dirty_list then clean_list to
  a private list, decrements `pages_owned`, releases the mutex, and frees the
  pages outside the lock. The new code should be that primitive with a
  different trigger.

## 3. Why "drain when the last client closes" is not enough

The obvious hook is `nv_close_device`, which stops the device when
`nvl->usage_count` reaches zero. On the Spark box that never happens while a
desktop session is up: `lsof` shows Xorg (15 fds), gnome-shell, mutter,
xdg-desktop-portal, snapd-desktop and nvidia-persistenced all holding
`/dev/nvidia0` open. `nvidia-persistenced` alone keeps `usage_count` above
zero on any headless box with persistence mode on, which is the normal
serving configuration.

So a hook gated on the device going idle would be inert exactly where we need
it. The trigger has to be **any client file close**, and the amount released
has to be governed by a watermark rather than by "no clients remain".

Pooled pages are, by definition, free memory. Releasing them costs only the
zeroing on the next allocation (the kernel zeroes `__GFP_ZERO` pages anyway;
the pool's benefit is pre-scrubbed pages and skipping the buddy allocator).
On a serving box with one long-lived engine that benefit is confined to model
load, so trading it for correct memory accounting is the right default here.

## 3b. Prior art upstream (checked 2026-09-05)

Searched NVIDIA/open-gpu-kernel-modules issues and pull requests for the
pool by name and by symptom (`EnableSystemMemoryPools`, `nv_mem_pool`,
shrinker, memory not released after exit, MemAvailable, DGX Spark, GB10,
GH200, 590 memory leak). Result: **nothing upstream addresses pool
retention.** No issue reports it and no PR trims or caps the pools. The only
public record of the symptom is the NVIDIA developer forum thread cited in
section 1, which was answered with "unsupported past 580" and no fix. This
change is new work, not a rebase of someone else's.

What does exist, and how it bears on the design:

- **PR #1004** "Fix use-after-free races in memory pool shrinker and DRM
  fence destruction" (neoyubi, Jan 2026, DRAFT, unmerged). It touches the
  same file: it moves `nv_mem_pool_shrinker_free` to the start of
  `nv_mem_pool_destroy`, NULLs `mem_pool->shrinker`, and adds
  `synchronize_rcu()` so a kswapd shrinker walk cannot race pool teardown.
  NVIDIA's maintainer asked for crash evidence; the author later attributed
  the crashes to a PCIe misconfiguration and could not reproduce, so it sits
  in draft. It concerns module-unload teardown only and does not change
  the scan callback or add trimming. Two implications for us: (a) the trim
  must never be called during or after `nv_destroy_page_pools`, which the
  close-callback trigger already guarantees since module exit stops the
  deferred-close queue first (nv.c:549 before nv.c:552); (b) if you factor
  the scan's list-moving into a shared helper, keep `nv_mem_pool_destroy`'s
  structure intact so a later rebase onto #1004 stays trivial.
  https://github.com/NVIDIA/open-gpu-kernel-modules/pull/1004
- **Issue #1269** "64K-page ARM64: DMA submap size is not 2MiB-aligned"
  (Jul 2026, open, patch attached, no NVIDIA response). GB10-specific and
  590+, but about `NV_DMA_SUBMAP_MAX_PAGES` on `CONFIG_ARM64_64K_PAGES`
  kernels, not about pools. Not applicable to the Spark box as configured:
  `getconf PAGESIZE` is 4096 there. Worth knowing if a 64K-page kernel is
  ever used. https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1269
- **Issue #613** "kernel memory not released" (535.161.07, cudaHostAlloc,
  kmemleak). Predates the pools (introduced in 590) and is a different
  path. Not related. https://github.com/NVIDIA/open-gpu-kernel-modules/issues/613
- Issue #999 mentions `EnableSystemMemoryPools: 529` only as part of a
  params dump in a D3cold report. Not related.

Also checked with the authenticated `gh` CLI on 2026-09-05: upstream's
newest tag is 610.57.04, the same base as this fork, and upstream `main`'s
`kernel-open/nvidia/nv-vm.c` and `nv-reg.h` carry no pool trimming, no
watermark regkey, and no change to the pool code relative to 610.57.04. So
there is nothing newer to rebase onto and nothing to wait for. Issue #848
("Driver fails to allocate memory for decoder", 575.x, discrete RTX 50xx VRAM
exhaustion) is the only hit for `drop_caches` and is unrelated.

Suggested follow-through once the patch is validated: open an upstream
issue with the section 5 measurements and the forum thread as the
symptom report, and the patch as a PR. Nothing there today gives NVIDIA a
reproducible account of the problem.

## 4. What to implement

### 4.1 A trim primitive in nv-vm.c

Add and export (declare in `nv-proto.h`):

```c
/*
 * Free pooled system pages until at most `retain_pages` order-0 pages remain
 * across all pools. Safe to call from any sleepable context; the scrubber
 * worker and the shrinker may run concurrently.
 */
void nv_trim_page_pools(unsigned long retain_pages);
```

Implementation sketch, per pool in `sysmem_page_pools[node][order]` that is
non-NULL:

1. Compute this pool's share to keep. Simplest correct rule: walk pools from
   highest order to lowest, keep entries until the running total of
   `pages_owned << order` reaches `retain_pages`, free everything beyond.
   With the default retain of 0 this degenerates to "free all", which is the
   common case.
2. Under `mem_pool->lock` (via `os_acquire_mutex`, which requires
   `NV_MAY_SLEEP()`), move entries from `dirty_list` first and then
   `clean_list` onto a local reclaim list with `nv_mem_pool_move_pages`,
   subtract from `pages_owned`, release the lock.
3. `nv_mem_pool_free_page_list(&reclaim_list, mem_pool->order)` outside the
   lock.
4. Print through `nv_printf(NV_DBG_MEMINFO, ...)` with node, order, freed and
   remaining counts, matching the existing messages in the file.

Do not stop the scrubber queue and do not touch `nv_mem_pool_destroy`. An
entry the scrubber is zeroing at that moment is on neither list, so it is
not touched; the worker re-adds it to clean_list on its next pass and it is
released by the next trim. `pages_owned` therefore can stay above zero by at
most one entry per pool after a full drain. That is fine; document it in a
comment.

Factor the list-moving and counting out of `nv_mem_pool_shrinker_scan` so
both callers share it rather than duplicating the mutex dance.

### 4.2 A regkey for the watermark

In `nv-reg.h`, next to `NVreg_EnableSystemMemoryPools`:

```
NVreg_SystemMemoryPoolRetainMB
  Amount of freed system memory, in MiB, the page pools may keep cached
  after a client closes its device file. 0 returns everything to the OS on
  every client close. 0xFFFFFFFF disables trimming on client close and keeps
  the pre-610.57 behavior (pools grow without bound until the kernel's
  shrinker runs). Default: 0.
```

Define it with `NV_DEFINE_REG_ENTRY_GLOBAL`, add the
`NV_DEFINE_PARAMS_TABLE_ENTRY`, declare `extern NvU32` in nv-vm.c the way
`NVreg_EnableSystemMemoryPools` is at nv-vm.c:29. Convert MiB to order-0
pages once (`(NvU64)mb << 20 >> PAGE_SHIFT`), guarding the sentinel.

Default 0 is deliberate. The pool's purpose is intra-process reuse; a client
close is the moment that purpose ends for that client's pages. If a
measurement later shows model-load time on Spark regressing meaningfully,
the operator can raise the watermark; the default should be the one that
makes memory accounting truthful.

### 4.3 The trigger in nv.c

Call the trim from `nvidia_close_callback` after `rm_cleanup_file_private`
has run and after `nvl->ldata_lock` has been released. Ordering matters:

- `rm_cleanup_file_private(sp, nv, &nvlfp->nvfp)` is where the RM tears down
  the client's allocations, which is what pushes the pages into the pools
  via `nv_free_system_pages`. Trimming before it would miss the client's
  own pages.
- The trim takes each pool's mutex and may sleep. Run it after
  `up(&nvl->ldata_lock)` in the normal (non-removal) branch, not while
  holding `ldata_lock`, and not in the surprise-removal branch that frees
  `nvl`.
- `nvidia_close_callback` runs either directly from `nvidia_close` or from
  `nvidia_close_deferred` on `nv_deferred_close_kthread_q`; both are
  sleepable, both hold `nv_system_pm_lock` for read. Nothing else is needed.

Gate it: skip entirely when the regkey holds the disable sentinel, and skip
when `sysmem_page_pools` has no pools (the `NVreg_EnableSystemMemoryPools=0`
case, see section 6), so the hook costs nothing where it has nothing to do.

Do not put the trim in `nv_close_device` under the `usage_count == 0` test;
section 3 explains why that never fires here.

### 4.4 Things not to change

- Leave `NVreg_EnableSystemMemoryPools` and its default alone. Operators who
  disable pools outright must keep working exactly as today.
- Leave the shrinker registered. Memory pressure between client closes
  still needs a release path, and `drop_caches` must keep working.
- No changes under `src/nvidia/` (the RM). This is purely the Linux glue
  layer's page cache, and the fix should stay in `kernel-open/`.

## 5. Acceptance

All on the Spark box, driver reloaded with the patched module. Record raw
numbers in the SlimServe notebook entry for 2026-09-05 (spark bring-up) or
hand them to whoever is running it.

1. **Baseline the phantom.** With the stock module: note `MemAvailable`;
   run a python that does `torch.empty(64 * 2**30, dtype=torch.uint8,
   device="cuda").fill_(1)` and exits; wait 10 s; note `MemAvailable`.
   Expected today: it stays roughly 64 GiB lower and `nvidia-smi` shows no
   process. This is the failing case.
2. **Same test with the patched module, default watermark.** `MemAvailable`
   must return to within 1 GiB of its pre-run value within 10 s of process
   exit, with the desktop session still running and nvidia-persistenced
   still holding the device. `grep -c nv_trim_page_pools` in `dmesg` with
   `NVreg_ResmanDebugLevel` raised, or the MEMINFO prints, should show the
   trim firing on that close.
3. **Watermark honored.** `NVreg_SystemMemoryPoolRetainMB=4096`: repeat; the
   retained amount after exit must be about 4 GiB, not 0 and not 64.
4. **Sentinel restores old behavior.** `NVreg_SystemMemoryPoolRetainMB=
   0xFFFFFFFF`: repeat; the phantom returns, proving the default path is the
   only thing that changed.
5. **Pools disabled composes.** `NVreg_EnableSystemMemoryPools=0` plus the
   patched module: no oops, no trim messages, memory returns immediately.
6. **No regression under churn.** Loop 20 iterations of allocate 32 GiB /
   free / exit while a second process holds a 16 GiB CUDA allocation and a
   third runs `stress-ng --vm 2 --vm-bytes 8G`, to exercise the trim racing
   the scrubber and the shrinker. No warnings in dmesg, `pages_owned` never
   goes negative (add a `WARN_ON` in the trim if it would).
7. **The actual failure.** From `~/SlimServe`: run any CUDA smoke test,
   exit, then immediately
   `SLIMSERVE_BRINGUP=1 .venv/bin/python -m slimserve.cli qwen38-nvfp4-1 --serve -y`.
   vLLM's startup memory check must pass without `drop_caches` and without
   waiting. (The profile is gated; the env var is the documented bring-up
   override.)

## 6. Relationship to the interim fix

Operators are pursuing the no-source-change option in parallel: a
`/etc/modprobe.d` file setting `options nvidia NVreg_EnableSystemMemoryPools=0`,
plus `update-initramfs -u` and a reboot because the nvidia modules load from
the initramfs on this box. That disables pool creation entirely, so
`sysmem_page_pools` is all NULL and the new trim is a no-op. The two are
independent and compose; the patched module is the version that keeps the
pool's reuse benefit for the running process while still returning memory at
exit, and it does not require the modprobe option.

Neither is repo configuration. Per SlimServe's deployment rule, modprobe.d
files, signing keys and install steps stay on the operator's machine and in
the notebook, never in the SlimServe repo.

## 7. Build, sign, install (as previously done on this box)

- Build: `make -C kernel-open -j$(nproc) modules` at the tree root builds
  the modules for the running kernel in about 35 s on the Spark
  (verified 2026-09-05 by the driver A/B session).
- Secure Boot is enabled (`mokutil --sb-state`), so the resulting `.ko`
  files must be signed with the enrolled MOK before they will load. The
  A/B session already has the signing step; reuse it.
- Install over the DKMS-provided 610.43.02 modules or via DKMS for
  610.57.04, then `update-initramfs -u`, then reboot or reload with gdm
  stopped. The installed driver is 610.43.02 and this tree is 610.57.04;
  the userspace on the box (libcuda from the 610.43 packages) must match
  whatever kernel module version ends up installed, so either build the
  patch against 610.43.02 sources or install the full 610.57.04 userspace.
  Say which one you did.

## 8. Deliverables

1. Commit(s) in this tree touching `kernel-open/nvidia/nv-vm.c`,
   `kernel-open/nvidia/nv.c`, `kernel-open/nvidia/nv-reg.h`,
   `kernel-open/common/inc/nv-proto.h`, with the regkey documented in
   nv-reg.h in the same style as the surrounding options.
2. The acceptance table from section 5 with raw before/after `MemAvailable`
   values and the driver version string from `/proc/driver/nvidia/version`.
3. A short note of anything you found that contradicts this document. The
   line numbers above are from `7493f4cd`; re-check them against whatever
   base you patch.
