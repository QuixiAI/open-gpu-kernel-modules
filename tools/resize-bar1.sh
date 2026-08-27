#!/bin/bash
# Resize BAR1 to its maximum supported size on every NVIDIA GPU, then reload
# the driver. For server boards whose firmware has no Resizable BAR option and
# allocates only the boot-time 256 MiB windows.
#
# What it does:
#   1. unloads the nvidia modules (anything using the GPUs must be stopped),
#   2. programs each GPU's Resizable BAR control register (BAR1 -> max
#      supported size) directly in config space,
#   3. removes the PCI subtree under each GPU's root port and rescans, so the
#      kernel re-enumerates the GPUs with the large BAR and builds fresh
#      bridge windows (an in-place resize via sysfs fails with ENOSPC because
#      the existing windows cannot be regrown around live siblings),
#   4. reloads the driver.
#
# Requirements: root; large MMIO aperture in firmware (see README);
# `pci=realloc` on the kernel command line is recommended.
# The ReBAR register resets on reboot -- run this every boot (systemd unit
# in tools/nvidia-resize-bar1.service).
set -u

echo "== stopping GPU users, unloading driver =="
systemctl stop nvidia-persistenced 2>/dev/null
fuser -k /dev/nvidia* 2>/dev/null
sleep 2
for m in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
  lsmod | grep -q "^$m " && rmmod $m
done
lsmod | grep -q "^nvidia " && { echo "FATAL: nvidia module still loaded"; exit 1; }

echo "== programming Resizable BAR control (BAR1 -> max supported) =="
python3 - <<'PY'
import glob, struct, sys

def rd32(f, off):
    f.seek(off); return struct.unpack('<I', f.read(4))[0]

def wr32(f, off, val):
    f.seek(off); f.write(struct.pack('<I', val))

gpus = []
for dev in sorted(glob.glob('/sys/bus/pci/devices/*')):
    try:
        vendor = open(dev + '/vendor').read().strip()
        cls = open(dev + '/class').read().strip()
    except OSError:
        continue
    if vendor == '0x10de' and cls.startswith('0x0300'):
        gpus.append(dev)
if not gpus:
    print('no NVIDIA VGA devices found'); sys.exit(1)

ok = True
for dev in gpus:
    name = dev.split('/')[-1]
    with open(dev + '/config', 'r+b') as f:
        # walk the extended capability chain for Resizable BAR (ID 0x0015)
        off, rebar, seen = 0x100, None, set()
        while off and off not in seen:
            seen.add(off)
            hdr = rd32(f, off)
            if hdr in (0, 0xFFFFFFFF):
                break
            if hdr & 0xFFFF == 0x0015:
                rebar = off
                break
            off = (hdr >> 20) & 0xFFC
        if rebar is None:
            print(name, 'NO REBAR CAPABILITY'); ok = False; continue
        cmd = rd32(f, 4) & 0xFFFF
        f.seek(4); f.write(struct.pack('<H', cmd & ~0x2))  # memory decode off
        nbars = (rd32(f, rebar + 8) >> 5) & 0x7
        done = False
        for i in range(nbars):
            cap_off = rebar + 4 + 8 * i
            ctl_off = rebar + 8 + 8 * i
            ctrl = rd32(f, ctl_off)
            if ctrl & 0x7 != 1:          # only the BAR1 entry
                continue
            sizes = rd32(f, cap_off) >> 4  # bit n = 2^n MiB supported
            want = sizes.bit_length() - 1  # largest supported size
            wr32(f, ctl_off, (ctrl & ~0x3F00) | (want << 8))
            back = rd32(f, ctl_off)
            done = (back >> 8) & 0x3F == want
            print(name, f'BAR1 -> {1 << want} MiB:', 'OK' if done else 'WRITE MISMATCH')
        f.seek(4); f.write(struct.pack('<H', cmd))  # restore decode
        if not done:
            ok = False
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo "ABORT: ReBAR programming failed, nothing removed"; exit 1; }

echo "== removing GPU root-port subtrees =="
ports=""
for dev in $(lspci -d 10de: -D | awk '$2 ~ /^(VGA|3D)/ {print $1}'); do
  p=$(readlink -f /sys/bus/pci/devices/$dev)
  port=$(echo "$p" | grep -o '[0-9a-f]\{4\}:[0-9a-f]\{2\}:[0-9a-f]\{2\}\.[0-9a-f]' | head -1)
  case " $ports " in *" $port "*) ;; *) ports="$ports $port";; esac
done
echo "root ports:$ports"
for port in $ports; do
  echo "  removing $port"
  echo 1 > /sys/bus/pci/devices/$port/remove
done
sleep 2

echo "== rescanning PCI bus =="
echo 1 > /sys/bus/pci/rescan
sleep 3

echo "== reloading driver =="
modprobe nvidia && modprobe nvidia_uvm && modprobe nvidia_modeset
systemctl start nvidia-persistenced 2>/dev/null
sleep 2
echo "== BAR1 sizes now =="
nvidia-smi -q | grep -A1 "BAR1 Memory" | grep Total
