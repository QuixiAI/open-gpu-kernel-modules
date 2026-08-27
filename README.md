# open-gpu-kernel-modules — P2P for GeForce (3090 / 4090 / 5090)

This is **QuixiAI / Eric Hartford's fork of NVIDIA's open GPU kernel
modules** (driver 610.57.04) that enables **PCIe peer-to-peer (P2P) between
consumer GeForce GPUs** — RTX 3090 (GA102), RTX 4090 (AD102), and RTX 5090
(GB202). It is inspired by and derived from
[George Hotz / tinygrad's P2P patch](https://github.com/tinygrad/open-gpu-kernel-modules)
(later simplified by aikitoria), ported forward to the 610 driver series.

NVIDIA's driver refuses P2P on GeForce boards. On a multi-GPU box that
means every byte exchanged between GPUs is staged through host RAM — and on
a typical EPYC/Threadripper host, concurrent device-to-host writes collapse
to a few GB/s aggregate no matter how many x16 links you have. With this
fork, GPUs DMA into each other directly over PCIe through BAR1:

```
$ nvidia-smi topo -p2p r
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
 GPU0   X       OK      OK      OK      OK      OK      OK      OK
 ...                                    (previously: GNS everywhere)
```

Measured on an 8x RTX 3090 EPYC host: direct peer copies at **23–26 GB/s**
across the root complex (PCIe 4.0 x16 line rate) versus ~2 GB/s per GPU
through host staging before, and NCCL all-reduce bus bandwidth of
**24.7 GB/s** versus 2.7 GB/s on NCCL's no-P2P SHM transport. Getting
there takes three ingredients — the patched modules, a full-size BAR1,
and one NCCL setting — all covered below; a P2P matrix showing OK is the
start, not the finish line.

## How it works

The 565-era tinygrad patch hand-built page tables to alias peer framebuffer
through BAR1. The 610 driver already ships NVIDIA's **native BAR1 P2P
implementation** end-to-end (external PTE building with the `SYS_NONCOH`
aperture and BAR1 DMA addressing in `nv_gpu_ops`, `UVM_GPU_LINK_PCIE_BAR1`
in `nvidia-uvm`, and a `PCIE_BAR1` connection type in the P2P API) — it is
simply never selected for GeForce. This fork is therefore a small,
surgical set of force-enables (one commit, four source files):

- `kernel-open/nvidia/nv-pci.c` — always attempt the resizable-BAR resize,
  so BAR1 grows to cover all of the framebuffer and static BAR1 mapping
  auto-enables. (On desktop boards with ReBAR enabled this succeeds or is
  already done by firmware; on server boards the in-place resize fails
  and `tools/resize-bar1.sh` finishes the job — see below.)
- `src/nvidia/src/kernel/gpu/bif/kernel_bif.c` — force the P2P read/write
  capabilities (`p2pOverride = 0x11`, bypassing the chipset allowlist) and
  default the PCIe P2P type to BAR1 (the modern replacement for the old
  `FORCE_P2P_TYPE_BAR1P2P` regkey, and exactly what
  `kbusIsPcieBar1P2PMappingSupported_GH100` checks).
- `src/nvidia/generated/g_kern_bus_nvoc.c` — route pre-Hopper chips to the
  chip-independent GH100 BAR1-P2P HAL implementations. This covers GA102
  and AD102; GB202 already uses the GH100 entries natively.
- `src/nvidia/src/kernel/gpu/bus/arch/pascal/kern_bus_gp100.c` — dispatch
  the `_PCIE_BAR1` connection type to the BAR1-P2P create/remove HALs,
  mirroring the GH100 dispatch.

No userspace changes: the modules pair with the stock 610.57.04 userspace
driver and GSP firmware.

## Requirements

- A 610.57.04 userspace driver install (check `nvidia-smi`; the kernel
  modules must match the userspace version exactly).
- Kernel headers for your running kernel.
- **BAR1 must end up covering the whole framebuffer** (32 GB on a 3090).
  This is not cosmetic: BAR1-P2P maps peer memory *through* the peer's
  BAR1 aperture. With the stock 256 MiB BAR1, `nvidia-smi topo -p2p r`
  still reports OK and plain `cudaMemcpyPeer` works (transient mappings),
  but **NCCL hangs at its first collective** — even with only 2 GPUs —
  because its persistent peer mappings don't fit. How to get the large
  BAR1 depends on your board; see
  [Getting the large BAR1](#getting-the-large-bar1) below.
- Kernel command line: add **`iommu=pt`** (or disable the IOMMU). With the
  IOMMU in translated mode we measured peer copies collapsing to
  0.9 GB/s on some pairs; with `iommu=pt` every pair ran at 23–26 GB/s.
  `pci=realloc` is also recommended on server boards.
- Linux x86_64 (aarch64 builds too; see NVIDIA's upstream docs for
  cross-compilation variables).

## Install

```bash
git clone https://github.com/QuixiAI/open-gpu-kernel-modules
cd open-gpu-kernel-modules
./install.sh
```

`install.sh` does: `rmmod` → `make modules -j$(nproc)` →
`make modules_install` → `depmod` → `nvidia-smi`.

Two gotchas we hit on a real box, so you don't have to:

1. **`rmmod` fails with "Module nvidia is in use".** Anything holding
   `/dev/nvidia*` keeps the old modules loaded and the install silently
   ends with the *old* driver still active (the version string is
   identical, so it looks fine). Stop every GPU process **and**
   `nvidia-persistenced` first:

   ```bash
   sudo systemctl stop nvidia-persistenced
   sudo fuser -k /dev/nvidia*        # or stop your serving/compute jobs
   sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia   # loaded subset
   ```

2. **A DKMS driver install shadows this one.** `modules_install` puts the
   patched modules in `kernel/drivers/video/`, but modprobe prefers
   `updates/dkms/`. If `modinfo -n nvidia` points at
   `/lib/modules/$(uname -r)/updates/dkms/`, move the DKMS copies aside:

   ```bash
   sudo mkdir -p /root/nvidia-dkms-backup
   sudo mv /lib/modules/$(uname -r)/updates/dkms/nvidia*.ko.zst /root/nvidia-dkms-backup/
   sudo depmod -a
   sudo modprobe nvidia nvidia_uvm nvidia_modeset
   sudo systemctl start nvidia-persistenced
   ```

Then get BAR1 to full size (next section) and verify.

## Getting the large BAR1

### Consumer boards (desktop BIOS)

Enable **Above 4G Decoding** and **Resizable BAR** in BIOS setup and
reboot. The firmware resizes BAR1 at POST, the windows are sized to
match, and the driver's own resize at probe is a no-op. Done.

### Server boards (no "Resizable BAR" BIOS item)

Server firmware (tested: Gigabyte MZ22-G20, EPYC Rome, 8x RTX 3090)
usually has no ReBAR toggle. It enumerates each GPU at the boot-time
256 MiB BAR1 and sizes every bridge window to match, so after boot you'll
see the driver fail like this in `dmesg`:

```
nvidia 0000:c3:00.0: BAR 1 [mem size 0x800000000 64bit pref]: can't assign; no space
```

Three things have to line up:

1. **Above 4G Decoding: Enabled** (usually already is).
2. **A large 64-bit MMIO aperture.** On AMD EPYC the item is
   *Prefetchable MMIO Above 4G Size* (Chipset → Fabric Resource, or
   AMD CBS → DF Common Options → Memory Addressing → MMIO High Size).
   You need at least `num_GPUs x max_BAR` per root complex the GPUs live
   behind; set it to the maximum (ours: 2 TB per NBIO). Verify from
   Linux — each root bridge should declare a huge window:

   ```
   $ dmesg | grep "root bus resource \[mem 0x"
   pci_bus 0000:c0: root bus resource [mem 0x10090200000-0x2bf53ffffff window]  # ~1.7 TB: good
   ```

3. **Re-enumerate the GPUs with the big BAR.** Even with a huge
   aperture, the firmware-sized 800 MiB bridge windows can't be regrown
   in place: the driver's resize at probe fails, and so does
   `echo 15 > /sys/bus/pci/devices/.../resource1_resize` (ENOSPC),
   because live sibling devices pin the existing windows. The fix is to
   program the Resizable BAR control register directly, drop the GPU
   subtrees, and rescan — on re-enumeration the GPUs advertise the
   32 GB BAR natively and the kernel builds fresh windows that fit.
   `tools/resize-bar1.sh` does the whole dance (stop GPU users → unload
   driver → program ReBAR on every NVIDIA GPU → remove each GPU root
   port → rescan → reload driver):

   ```bash
   sudo ./tools/resize-bar1.sh
   ...
   == BAR1 sizes now ==
       Total                                          : 32768 MiB   (x8)
   ```

   The ReBAR register resets at reboot and the firmware goes back to
   256 MiB, so run it every boot:

   ```bash
   sudo cp tools/resize-bar1.sh /usr/local/sbin/
   sudo cp tools/nvidia-resize-bar1.service /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl enable nvidia-resize-bar1
   ```

## NCCL on many-GPU PCIe boxes

Two settings decide whether NCCL actually uses the fast path:

- **`NCCL_P2P_LEVEL=SYS`** — by default NCCL only uses P2P between GPUs
  under the same PCIe switch and falls back to its SHM transport for
  pairs whose path crosses the CPU ("SYS" paths). On a multi-root-complex
  box that silently caps you near SHM speed.
- Do **not** set `NCCL_P2P_DISABLE=1` (remove it if you added it while
  the box had no P2P).

Measured on 8x RTX 3090 (EPYC Rome, four root complexes, PCIe 4.0 x16),
all-reduce bus bandwidth at 84 MB payloads:

| Configuration                          | busbw       | small-AR latency |
|----------------------------------------|-------------|------------------|
| No P2P (NCCL SHM transport)            | 2.7 GB/s    | —                |
| P2P, default `NCCL_P2P_LEVEL`          | 4.2 GB/s    | 41 us            |
| P2P, `NCCL_P2P_LEVEL=SYS`              | **24.7 GB/s** | **41 us**      |

## Verify

```bash
# BAR1 should be ~32 GiB (not 256 MiB) after the resize:
nvidia-smi -q -i 0 | grep -A3 "BAR1 Memory"

# The P2P read matrix should be all OK:
nvidia-smi topo -p2p r

# And from PyTorch:
python -c "import torch; print(torch.cuda.can_device_access_peer(0, 1))"

# The real acid test — an actual NCCL collective (hangs if BAR1 is small):
NCCL_P2P_LEVEL=SYS torchrun --nproc-per-node=2 - <<'EOF'
import torch, torch.distributed as dist, os
dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["RANK"]))
x = torch.ones(1 << 20, device="cuda")
dist.all_reduce(x)
print("all_reduce OK:", x[0].item() == dist.get_world_size())
EOF
```

A quick bandwidth check (adjust devices to a pair on your box):

```bash
python - <<'EOF'
import time, torch
a = torch.randn(2**25, device="cuda:0")   # 128 MiB
b = torch.empty_like(a, device="cuda:1")
for _ in range(3): b.copy_(a)
torch.cuda.synchronize(0); torch.cuda.synchronize(1)
t0 = time.perf_counter()
for _ in range(10): b.copy_(a, non_blocking=True)
torch.cuda.synchronize(0); torch.cuda.synchronize(1)
print(f"{a.numel()*4*10/(time.perf_counter()-t0)/1e9:.1f} GB/s")
EOF
```

Expect roughly PCIe line rate (~20+ GB/s on Gen4 x16) between peers.

## Rollback

The stock modules are whatever your distro/DKMS installed; if you moved
them aside per gotcha 2, restore with:

```bash
sudo mv /root/nvidia-dkms-backup/nvidia*.ko.zst /lib/modules/$(uname -r)/updates/dkms/
sudo depmod -a && sudo reboot
```

## Building manually

```bash
make modules -j$(nproc)
sudo make modules_install -j$(nproc)
sudo depmod
```

The modules must be used with GSP firmware and userspace components from
the matching 610.57.04 release. To install the userspace driver without
its own kernel modules: `sh ./NVIDIA-Linux-[...].run --no-kernel-modules`.
For everything else (supported architectures, kernel compatibility, the
full module set), see
[NVIDIA's upstream repository](https://github.com/NVIDIA/open-gpu-kernel-modules)
— this fork tracks it with a single commit on top.

## Credits

- **George Hotz / tinygrad** — the original 565.57.01 P2P patch that
  proved consumer-GPU P2P was a driver policy, not a hardware limit.
- **aikitoria** — the simplified port this fork's commit is based on.
- **QuixiAI / Eric Hartford** — the 610.57.04 port: dropped the manual
  PTE/aperture hacks in favor of the driver's now-native BAR1-P2P path,
  extended coverage to GA102/AD102/GB202, and validated on 8x RTX 3090.

## Disclaimer

This modifies driver behavior NVIDIA explicitly disables on GeForce
hardware. It works on the boards listed above with ReBAR-capable
platforms; anything else is uncharted. No warranty — validate P2P
correctness on your own workload (a bandwidth test plus an all-reduce
correctness check is a good minimum).
