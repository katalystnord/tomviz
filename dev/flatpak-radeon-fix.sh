#!/bin/sh
# Make the official Tomviz flatpak (org.tomviz.Tomviz) run on GNOME Wayland
# with a recent AMD GPU. Tested with Tomviz 3.1.0 on a Radeon RX 9070.
#
# Two problems in the 3.1.0 flatpak:
# 1. The manifest uses fallback-x11, so the sandbox gets no X11 on Wayland,
#    but the bundled Qt tries xcb first and aborts. Fix: force Qt to wayland.
# 2. The launcher prepends the bundled conda lib/ to LD_LIBRARY_PATH, so the
#    conda libdrm (2.4.129) shadows the one the Mesa GL extension was built
#    against (2.4.134), and Mesa fails with "ac_query_gpu_info failed".
#    Fix: preload the GL extension's own libdrm.
#
# Usage: dev/flatpak-radeon-fix.sh [apply|reset|show]   (default: apply)

set -eu
APP=org.tomviz.Tomviz
GL=/usr/lib/x86_64-linux-gnu/GL/default/lib

case "${1:-apply}" in
  apply)
    flatpak override --user \
      --env=QT_QPA_PLATFORM=wayland \
      --env=LD_PRELOAD="$GL/libdrm.so.2:$GL/libdrm_amdgpu.so.1" \
      "$APP"
    flatpak override --user --show "$APP"
    ;;
  reset)
    flatpak override --user --reset "$APP"
    ;;
  show)
    flatpak override --user --show "$APP"
    ;;
  *)
    echo "usage: $0 [apply|reset|show]" >&2
    exit 1
    ;;
esac
