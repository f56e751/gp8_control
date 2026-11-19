"""[공통 로봇] GP8 운동학 + 하드웨어 실행 심(seam).

gp8.py / base_robot.py / lie_algebra.py   PoE FK/IK
robot_base.py      RobotBackend: 500 Hz 스트림 엔진, 격자시간 석션 릴리즈 (HW/sim 공용 로직)
trajectory_controller.py  TrajectoryController(RobotBackend): JGPC publish + 석션 TCP 50242
world_base.py / ros_world.py  WorldSource: /camera_debug/detections + 벨트 엔코더
motion_logger.py   진단 CSV
"""
from gp8_control.robot.robot_base import RobotBackend
from gp8_control.robot.world_base import WorldSource

__all__ = ["RobotBackend", "WorldSource"]
