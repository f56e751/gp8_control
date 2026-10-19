# Source in a shell to use this workspace:   source <ws>/src/gp8_control/scripts/env.sh
# (sources ROS 2 Humble + THIS workspace's install; prints which gp8_control is active)
_GP8_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GP8_WS="$(cd "$_GP8_SCRIPTS/../../.." && pwd)"
export GP8_WS
source /opt/ros/humble/setup.bash
source "$GP8_WS/install/setup.bash"
echo "[gp8] workspace $GP8_WS -> $(ros2 pkg prefix gp8_control 2>/dev/null)"
