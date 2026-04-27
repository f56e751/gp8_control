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
        f"{package_name}.utils",
        f"{package_name}.trajectory",
        f"{package_name}.model",
        f"{package_name}.mock",
        f"{package_name}.gui",
        f"{package_name}.robots",
        f"{package_name}.tracking",
        f"{package_name}.planning",
        f"{package_name}.tests",
    ],
    package_dir={
        package_name: ".",
        f"{package_name}.controllers": "controllers",
        f"{package_name}.perception": "perception",
        f"{package_name}.utils": "utils",
        f"{package_name}.trajectory": "trajectory",
        f"{package_name}.model": "model",
        f"{package_name}.mock": "mock",
        f"{package_name}.gui": "gui",
        f"{package_name}.robots": "robots",
        f"{package_name}.tracking": "tracking",
        f"{package_name}.planning": "planning",
        f"{package_name}.tests": "tests",
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
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
            "camera_info_publisher = gp8_control.perception.camera_info_publisher:main",
            "mock_robot = gp8_control.mock.mock_robot:main",
            "gui_server = gp8_control.gui.server:main",
            "name_bridge = gp8_control.bridge:main",
            "terminal_debug = gp8_control.terminal_debug:main",
            "queue_test = gp8_control.tests.queue_test:main",
        ],
    },
)
