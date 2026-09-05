#!/bin/bash
# Build, sign (when Secure Boot needs it), install and reload the modules.
#
# Order matters: build first, so a build failure leaves the running driver
# untouched; unload only once there is something to install; never report
# success while the old modules are still loaded.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

KREL=$(uname -r)
SIGN_FILE=/lib/modules/$KREL/build/scripts/sign-file
MOK_PRIV=${MOK_PRIV:-/var/lib/shim-signed/mok/MOK.priv}
MOK_DER=${MOK_DER:-/var/lib/shim-signed/mok/MOK.der}

sb_enabled() { mokutil --sb-state 2>/dev/null | grep -q 'SecureBoot enabled'; }
mok_enrolled() { mokutil --test-key "$MOK_DER" 2>/dev/null | grep -q 'is already enrolled'; }

echo "== build =="
make modules -j"$(nproc)"

if sb_enabled; then
  echo "== Secure Boot is enabled: signing =="
  if [ ! -r "$MOK_PRIV" ] && ! sudo test -r "$MOK_PRIV"; then
    echo "ABORT: Secure Boot is on and no MOK private key at $MOK_PRIV." >&2
    echo "       Unsigned modules will not load; the box would come up GPU-less." >&2
    echo "       Create one (e.g. 'sudo update-secureboot-policy --new-key') or set MOK_PRIV/MOK_DER." >&2
    exit 1
  fi
  if ! mok_enrolled; then
    echo "ABORT: $MOK_DER is not enrolled in the firmware; signed modules would still be rejected." >&2
    echo "       Run: sudo mokutil --import $MOK_DER ; reboot ; 'Enroll MOK' in MokManager. Then rerun." >&2
    exit 1
  fi
  for ko in kernel-open/*.ko; do
    sudo "$SIGN_FILE" sha256 "$MOK_PRIV" "$MOK_DER" "$ko"
    modinfo -F signer "$ko" | grep -q . || { echo "ABORT: signing $ko failed" >&2; exit 1; }
  done
fi

echo "== unload =="
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
for m in nvidia_peermem nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
  if lsmod | grep -q "^$m "; then sudo rmmod "$m"; fi
done
if lsmod | grep -q '^nvidia '; then
  echo "ABORT: nvidia is still loaded (something holds /dev/nvidia*). Stop GPU users and rerun." >&2
  echo "       sudo fuser -v /dev/nvidia*" >&2
  exit 1
fi

echo "== install =="
sudo make modules_install -j"$(nproc)"
sudo depmod -a "$KREL"

echo "== reload =="
sudo modprobe nvidia && sudo modprobe nvidia_uvm && sudo modprobe nvidia_modeset && sudo modprobe nvidia_drm
sudo systemctl start nvidia-persistenced 2>/dev/null || true

echo "== verify =="
loaded=$(modinfo -n nvidia)
built=$(modinfo -F version kernel-open/nvidia.ko)
running=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -1)
echo "module file: $loaded"
echo "built $built, running $running"
case "$loaded" in */updates/dkms/*) echo "WARNING: a DKMS copy shadows this install (see README gotcha 2)";; esac
[ "$built" = "$running" ] || { echo "ABORT: running version differs from the build" >&2; exit 1; }
nvidia-smi
