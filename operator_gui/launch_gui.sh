#!/usr/bin/env bash
# Launch the CNC operator GUI with ROS and workspace already sourced.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROS_SETUP=""
for distro in jazzy iron humble; do
  if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
    ROS_SETUP="/opt/ros/${distro}/setup.bash"
    break
  fi
done

if [[ -z "${ROS_SETUP}" ]]; then
  echo "ERROR: Could not find ROS under /opt/ros" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${ROS_SETUP}"

if [[ -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
  # shellcheck source=/dev/null
  source "${WORKSPACE_ROOT}/install/setup.bash"
fi

cd "${SCRIPT_DIR}"
exec python3 cnc_operator_gui.py "$@"
