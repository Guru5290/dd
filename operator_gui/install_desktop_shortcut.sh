#!/usr/bin/env bash
# Install a desktop shortcut for the CNC Operator GUI on Ubuntu 24.04.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/launch_gui.sh"
ICON="${SCRIPT_DIR}/assets/cnc-operator-icon.svg"
DESKTOP_NAME="CNC-Operator-GUI.desktop"

DESKTOP_DIR="${HOME}/Desktop"
if [[ ! -d "${DESKTOP_DIR}" ]]; then
  DESKTOP_DIR="${HOME}/desktop"
fi
if [[ ! -d "${DESKTOP_DIR}" ]]; then
  mkdir -p "${HOME}/Desktop"
  DESKTOP_DIR="${HOME}/Desktop"
fi

TARGET_DESKTOP="${DESKTOP_DIR}/${DESKTOP_NAME}"
LOCAL_DESKTOP="${SCRIPT_DIR}/${DESKTOP_NAME}"

cat > "${LOCAL_DESKTOP}" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=CNC Operator GUI
GenericName=CNC Perception Panel
Comment=Launch the CNC perception operator control panel
Exec=${LAUNCHER}
Icon=${ICON}
Path=${SCRIPT_DIR}
Terminal=false
StartupNotify=true
Categories=Development;Engineering;Robotics;
Keywords=CNC;ROS;Perception;Operator;
EOF

chmod +x "${LAUNCHER}"
chmod +x "${LOCAL_DESKTOP}"
cp "${LOCAL_DESKTOP}" "${TARGET_DESKTOP}"
chmod +x "${TARGET_DESKTOP}"

if command -v gio >/dev/null 2>&1; then
  gio set "${TARGET_DESKTOP}" metadata::trusted true 2>/dev/null || true
fi

echo "Desktop shortcut installed:"
echo "  ${TARGET_DESKTOP}"
echo
echo "Double-click 'CNC Operator GUI' on your desktop to launch."
echo "If Ubuntu asks, choose 'Trust and Launch'."
