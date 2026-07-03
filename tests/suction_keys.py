#!/usr/bin/env python3
"""Standalone keyboard suction toggle for the adv4ncr driver (TCP 50242 Simple Message).

The adv4ncr RT driver exposes NO ``/write_single_io`` ROS service (unlike MotoROS2), so
``terminal_debug``'s space-bar suction does not work on it. This talks DIRECTLY to the
controller's ros-industrial Simple Message **IoServer on TCP 50242** — the same path
``TrajectoryController.suction_on/off`` uses (``controllers/trajectory_controller.py::
_call_io``) — so it needs no ROS, no launch, no MoveIt: just the robot reachable on the LAN.

Keys:
  space / s : toggle suction ON <-> OFF
  o         : suction ON
  f         : suction OFF
  q / Ctrl-C: quit (releases suction OFF first)

Env:
  GP8_ROBOT_IP    controller IP        (default 192.168.255.1)
  GP8_SUCTION_IO  suction OUT address  (default 10017)

Run (any of):
  python3 ~/ros2_ws/src/gp8_control/tests/suction_keys.py
  PYTHONPATH=~/ros2_ws/src python3 -m gp8_control.tests.suction_keys
  ros2 run gp8_control suction_keys          # after a colcon build

WARNING: the TCP 50242 IO write is HW-UNVERIFIED. Confirm on the pendant that the suction
OT actually toggles; run supervised. Value convention is 0 = ON, 1 = OFF (Schmalz ejector,
per the controller I/O map) — flip ON/OFF below if your ejector is wired the other way.
"""
from __future__ import annotations

import os
import socket
import struct
import sys

IP = os.environ.get("GP8_ROBOT_IP", "192.168.255.1")
PORT = 50242                       # TCP_PORT_IO (Controller.h)
ADDR = int(os.environ.get("GP8_SUCTION_IO", "10017"))
WRITE_IO_BIT = 2005               # ROS_MSG_MOTO_WRITE_IO_BIT
ON, OFF = 0, 1                    # value convention: 0 = ON, 1 = OFF


class SuctionIO:
    """Persistent TCP client to the controller's Simple Message IoServer (port 50242)."""

    def __init__(self, ip: str = IP, port: int = PORT) -> None:
        self.ip = ip
        self.port = port
        self.sock: "socket.socket | None" = None

    def write_bit(self, address: int, value: int) -> "int | None":
        """Write one IO bit; return the reply resultCode, or None on failure/short reply.

        Wire format (little-endian, packed), identical to trajectory_controller._call_io:
          prefix int32 = 20 (= len(header+body));
          header: msgType int32 = 2005, commType int32 = 2 (SERVICE_REQUEST), replyType int32 = 0;
          body:   ioAddress uint32, ioValue uint32.
        Reply: prefix(4) + header(12) + resultCode int32 @ offset 16.
        """
        pkt = struct.pack("<iiiiII", 20, WRITE_IO_BIT, 2, 0, int(address), int(value))
        for attempt in (0, 1):                    # one reconnect retry
            try:
                if self.sock is None:
                    self.sock = socket.create_connection((self.ip, self.port), timeout=2.0)
                    self.sock.settimeout(2.0)
                self.sock.sendall(pkt)
                reply = self.sock.recv(64)
                if len(reply) >= 20:
                    return struct.unpack_from("<i", reply, 16)[0]
                return None                       # connected but short/no reply
            except OSError as e:
                self._close()
                if attempt == 1:
                    print(f"  [IO] write failed to {self.ip}:{self.port}: {e}", file=sys.stderr)
                    return None
        return None

    def _close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    close = _close


def _make_getch():
    """Return a blocking single-keypress reader (raw terminal; POSIX or Windows)."""
    try:                                           # Windows
        import msvcrt

        return lambda: msvcrt.getch().decode("utf-8", errors="ignore")
    except ImportError:                            # POSIX
        import termios
        import tty

        def getch() -> str:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                return sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        return getch


def main() -> None:
    io = SuctionIO()
    getch = _make_getch()
    state_on = False

    def apply(on: bool) -> None:
        nonlocal state_on
        rc = io.write_bit(ADDR, ON if on else OFF)
        state_on = on
        rc_s = "NO REPLY (unreachable?)" if rc is None else f"resultCode={rc}"
        print(f"  suction {'ON ' if on else 'OFF'}  (addr {ADDR}, val {ON if on else OFF}) -> {rc_s}")

    print(f"Suction keyboard control -> {IP}:{PORT} (Simple Message IoServer), addr {ADDR}")
    print("  [space]/[s] toggle    [o] ON    [f] OFF    [q] quit\n")
    apply(False)                                   # start from a known-safe (OFF) state
    try:
        while True:
            key = getch()
            if key in (" ", "s", "S"):
                apply(not state_on)
            elif key in ("o", "O"):
                apply(True)
            elif key in ("f", "F"):
                apply(False)
            elif key in ("q", "Q", "\x03", "\x04"):    # q / Ctrl-C / Ctrl-D
                break
    except KeyboardInterrupt:
        pass
    finally:
        apply(False)                               # safety: always release on exit
        io.close()
        print("\nsuction released, socket closed. bye.")


if __name__ == "__main__":
    main()
