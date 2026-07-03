from setuptools import setup, find_packages
import os
from glob import glob

package_name = "gp8_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[
        package_name,
        f"{package_name}.controllers",
        f"{package_name}.perception",
        f"{package_name}.conveyor",
        f"{package_name}.utils",
        f"{package_name}.trajectory",
        f"{package_name}.model",
        f"{package_name}.mock",
        f"{package_name}.gui",
        f"{package_name}.robots",
        f"{package_name}.tracking",
        f"{package_name}.planning",
        f"{package_name}.skills",
        f"{package_name}.tests",
    ],
    package_dir={
        package_name: ".",
        f"{package_name}.controllers": "controllers",
        f"{package_name}.perception": "perception",
        f"{package_name}.conveyor": "conveyor",
        f"{package_name}.utils": "utils",
        f"{package_name}.trajectory": "trajectory",
        f"{package_name}.model": "model",
        f"{package_name}.mock": "mock",
        f"{package_name}.gui": "gui",
        f"{package_name}.robots": "robots",
        f"{package_name}.tracking": "tracking",
        f"{package_name}.planning": "planning",
        f"{package_name}.skills": "skills",
        f"{package_name}.tests": "tests",
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    package_data={
        f"{package_name}.gui": ["static/*", "static/css/*"],
        f"{package_name}.model": ["*.pt"],
    },
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="minu",
    maintainer_email="fortriver54321@gmail.com",
    description="Real robot control for Yaskawa GP8 pick-and-throw system",
    license="MIT",
    entry_points={
        "console_scripts": [
            "gp8_app = gp8_control.app:main",
            "mock_robot = gp8_control.mock.mock_robot:main",
            # MuJoCo-backed twin (Level B). NOTE: needs `mujoco` in the uv venv,
            # so launch it with .venv/bin/python -m gp8_control.mock.mujoco_robot
            # (or sim_mujoco.launch.py) — plain `ros2 run` uses system python.
            "mujoco_robot = gp8_control.mock.mujoco_robot:main",
            "fake_belt = gp8_control.mock.fake_belt:main",
            "gui_server = gp8_control.gui.server:main",
            "name_bridge = gp8_control.bridge:main",
            "terminal_debug = gp8_control.terminal_debug:main",
            "belt_viz = gp8_control.belt_viz:main",
            "camera_debug = gp8_control.camera_debug:main",
            "queue_test = gp8_control.tests.queue_test:main",
            # Stage-C persistent-queue HW gate (no torch; system python OK).
            "persistent_queue_spike = "
            "gp8_control.tests.persistent_queue_spike:main",
            # Measure the pure controller startup floor via FJT (no per-point push).
            "measure_fjt_floor = gp8_control.tests.measure_fjt_floor:main",
            # Same, via the QUEUE path (re-entry + per-point push + startup).
            "measure_queue_floor = gp8_control.tests.measure_queue_floor:main",
            # Standalone keyboard suction toggle for the adv4ncr driver (TCP 50242
            # Simple Message IoServer; no ROS service needed). Run supervised.
            "suction_keys = gp8_control.tests.suction_keys:main",
        ],
    },
)
