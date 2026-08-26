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

Measured on an 8x RTX 3090 EPYC host immediately after loading the modules:
direct peer copies at **~21 GB/s** across the root complex (PCIe 4.0 x16
line rate) versus ~2 GB/s per GPU through host staging before. Tensor
parallelism, NCCL, and vLLM/SGLang custom all-reduce paths light up
automatically once P2P reports OK.

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
  auto-enables.
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
- **Resizable BAR enabled in SBIOS** ("Above 4G Decoding" + "Resizable
  BAR"). The GPUs advertise ReBAR up to 32 GB; the driver resizes BAR1 at
  probe. A reboot after installing the modules is expected before the
  large BAR takes effect — with the stock 256 MiB BAR1, P2P reports OK but
  bandwidth between some GPU pairs is severely degraded.
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

Then **reboot** so the BAR1 resize applies during PCI probe.

## Verify

```bash
# BAR1 should be ~32 GiB (not 256 MiB) after the reboot:
nvidia-smi -q -i 0 | grep -A3 "BAR1 Memory"

# The P2P read matrix should be all OK:
nvidia-smi topo -p2p r

# And from PyTorch:
python -c "import torch; print(torch.cuda.can_device_access_peer(0, 1))"
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
