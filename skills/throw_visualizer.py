"""RViz markers for the trajectory actually dispatched by ``ThrowSkill``.

The ballistic preview deliberately starts at the configured suction-OFF
waypoint.  It therefore shows how changing ``RELEASE_LEAD`` changes the
predicted landing point.  Pneumatic lag, drag, object rotation, and collisions
with the bin are not modelled; the object is a point mass under gravity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import Point
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


GRAVITY = 9.81


@dataclass(frozen=True)
class ReleaseCandidate:
    """Ballistic result when one trajectory waypoint is used for release."""

    index: int
    eta: float
    time: float
    position: np.ndarray
    velocity: np.ndarray
    landing: np.ndarray
    flight_time: float
    error_xy: float


@dataclass(frozen=True)
class ThrowEvaluation:
    """Selected-release result plus the best release available on the arc."""

    selected: ReleaseCandidate
    best: ReleaseCandidate
    feasible_count: int
    goal_xy: np.ndarray
    goal_radius: float
    eta_index: int
    release_lead: float
    recommended_release_lead: float

    @property
    def selected_hit(self) -> bool:
        return self.selected.error_xy <= self.goal_radius

    @property
    def feasible_exists(self) -> bool:
        return self.feasible_count > 0


def ballistic_trajectory(
    position: np.ndarray,
    velocity: np.ndarray,
    impact_z: float,
    *,
    gravity: float = GRAVITY,
    samples: int = 80,
) -> tuple[np.ndarray, float]:
    """Return point-mass flight samples until the descending impact plane."""
    p0 = np.asarray(position, dtype=float).reshape(3)
    v0 = np.asarray(velocity, dtype=float).reshape(3)
    dz = float(p0[2] - impact_z)
    discriminant = float(v0[2] ** 2 + 2.0 * gravity * dz)
    if gravity <= 0.0 or discriminant < 0.0:
        raise ValueError("ballistic impact plane is unreachable")

    # Positive root of z0 + vz*t - 1/2*g*t^2 = impact_z.  This is the
    # descending intersection even when the initial velocity points down.
    flight_time = (float(v0[2]) + math.sqrt(discriminant)) / gravity
    if flight_time <= 0.0:
        raise ValueError("ballistic impact occurs at or before release")

    times = np.linspace(0.0, flight_time, max(2, int(samples)))
    points = p0[None, :] + times[:, None] * v0[None, :]
    points[:, 2] -= 0.5 * gravity * times * times
    points[-1, 2] = float(impact_z)
    return points, flight_time


class ThrowVisualizer:
    """Publish a latched marker set for the most recently planned throw."""

    TOPIC = "/gp8_manager/throw_preview"

    def __init__(
        self,
        node,
        robot,
        impact_z: float = 0.0,
        goal_xy=(1.1, -0.25),
        goal_radius: float = 0.10,
    ) -> None:
        self._node = node
        self._robot = robot
        self._impact_z = float(impact_z)
        self._goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
        self._goal_radius = float(goal_radius)
        if self._goal_radius <= 0.0:
            raise ValueError("throw goal radius must be positive")
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._publisher = node.create_publisher(MarkerArray, self.TOPIC, qos)

    @staticmethod
    def _point(xyz) -> Point:
        point = Point()
        point.x, point.y, point.z = map(float, xyz)
        return point

    def _marker(self, marker_id: int, namespace: str, marker_type: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self._node.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _color(marker: Marker, rgba) -> None:
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba

    def _evaluate_candidate(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        timestamp: float,
        index: int,
        eta_denominator: int,
    ) -> tuple[ReleaseCandidate, np.ndarray]:
        transform = self._robot.forward_kinematics(q)
        position = np.asarray(transform[:3, 3], dtype=float)
        twist = self._robot.jacobian(q) @ qd
        velocity = np.asarray(twist[3:] + np.cross(twist[:3], position), dtype=float)
        arc, flight_time = ballistic_trajectory(position, velocity, self._impact_z)
        landing = np.asarray(arc[-1], dtype=float)
        error_xy = float(np.linalg.norm(landing[:2] - self._goal_xy))
        candidate = ReleaseCandidate(
            index=int(index),
            eta=float(index) / float(max(1, eta_denominator)),
            time=float(timestamp),
            position=position,
            velocity=velocity,
            landing=landing,
            flight_time=float(flight_time),
            error_xy=error_xy,
        )
        return candidate, arc

    def publish(
        self,
        trajectory: np.ndarray,
        velocity: np.ndarray,
        timestamps: np.ndarray,
        release_index: int,
        eta_index: int,
        release_lead: float,
        throw_last_index: int,
        eta_min: float,
        eta_max: float,
        trajectory_hz: float,
    ) -> ThrowEvaluation | None:
        """Publish and return selected/best ballistic release evaluation."""
        try:
            q = np.asarray(trajectory, dtype=float)
            qd = np.asarray(velocity, dtype=float)
            ts = np.asarray(timestamps, dtype=float).reshape(-1)
            if q.ndim != 2 or qd.shape != q.shape or q.shape[1] != ts.size:
                raise ValueError(
                    f"trajectory shape mismatch q={q.shape}, qd={qd.shape}, ts={ts.shape}"
                )
            release_index = int(np.clip(release_index, 0, ts.size - 1))
            eta_index = int(np.clip(eta_index, 0, ts.size - 1))
            throw_last_index = int(np.clip(throw_last_index, 1, ts.size - 1))

            tcp_path = np.asarray([
                self._robot.forward_kinematics(q[:, i])[:3, 3]
                for i in range(q.shape[1])
            ])
            selected, arc = self._evaluate_candidate(
                q[:, release_index], qd[:, release_index], ts[release_index],
                release_index, throw_last_index,
            )
            release_pos = selected.position
            release_velocity = selected.velocity
            flight_time = selected.flight_time
            landing = selected.landing

            # Search only the learned throw arc and only the eta interval the
            # production decoder permits.  The appended home/standby chain is
            # not a valid throw-release region.
            first_candidate = max(0, int(math.ceil(float(eta_min) * throw_last_index)))
            last_candidate = min(
                throw_last_index,
                int(math.floor(float(eta_max) * throw_last_index)),
            )
            candidates: list[ReleaseCandidate] = []
            for index in range(first_candidate, last_candidate + 1):
                try:
                    candidate, _ = self._evaluate_candidate(
                        q[:, index], qd[:, index], ts[index], index, throw_last_index,
                    )
                    candidates.append(candidate)
                except ValueError:
                    continue
            if not candidates:
                raise ValueError("no ballistic release candidate reaches the impact Z plane")
            best = min(candidates, key=lambda item: item.error_xy)
            feasible_count = sum(
                candidate.error_xy <= self._goal_radius for candidate in candidates
            )
            recommended_lead = (
                float(eta_index - best.index) / float(trajectory_hz)
            )
            evaluation = ThrowEvaluation(
                selected=selected,
                best=best,
                feasible_count=feasible_count,
                goal_xy=self._goal_xy.copy(),
                goal_radius=self._goal_radius,
                eta_index=eta_index,
                release_lead=float(release_lead),
                recommended_release_lead=recommended_lead,
            )

            delete_all = Marker()
            delete_all.action = Marker.DELETEALL
            markers = [delete_all]

            # Actual commanded TCP path.  Pre-release is magenta; the remaining
            # robot follow-through/park is yellow.
            for marker_id, namespace, points, rgba, width in (
                (1, "tcp_before_release", tcp_path[:release_index + 1],
                 (1.0, 0.10, 0.75, 0.95), 0.014),
                (2, "tcp_after_release", tcp_path[release_index:],
                 (1.0, 0.80, 0.05, 0.75), 0.009),
            ):
                path = self._marker(marker_id, namespace, Marker.LINE_STRIP)
                path.scale.x = width
                self._color(path, rgba)
                path.points = [self._point(p) for p in points]
                markers.append(path)

            release = self._marker(3, "release", Marker.SPHERE)
            release.pose.position = self._point(release_pos)
            release.scale.x = release.scale.y = release.scale.z = 0.065
            self._color(release, (1.0, 0.25, 0.05, 0.98))
            markers.append(release)

            velocity_scale_s = 0.20
            arrow = self._marker(4, "release_velocity", Marker.ARROW)
            arrow.points = [
                self._point(release_pos),
                self._point(release_pos + velocity_scale_s * release_velocity),
            ]
            arrow.scale.x = 0.014
            arrow.scale.y = 0.030
            arrow.scale.z = 0.040
            self._color(arrow, (1.0, 0.35, 0.05, 0.98))
            markers.append(arrow)

            ballistic = self._marker(5, "ballistic", Marker.LINE_STRIP)
            ballistic.scale.x = 0.012
            self._color(ballistic, (0.0, 0.80, 1.0, 0.98))
            ballistic.points = [self._point(p) for p in arc]
            markers.append(ballistic)

            impact = self._marker(6, "predicted_impact", Marker.SPHERE)
            impact.pose.position = self._point(landing)
            impact.scale.x = impact.scale.y = impact.scale.z = 0.085
            self._color(
                impact,
                (0.10, 0.95, 0.20, 0.98) if evaluation.selected_hit
                else (1.0, 0.12, 0.08, 0.98),
            )
            markers.append(impact)

            release_text = self._marker(7, "release_text", Marker.TEXT_VIEW_FACING)
            release_text.pose.position = self._point(release_pos + np.array([0, 0, 0.10]))
            release_text.scale.z = 0.045
            self._color(release_text, (1.0, 1.0, 1.0, 1.0))
            release_text.text = (
                f"release t={ts[release_index]:.3f}s  idx={release_index}"
                f"  lead={release_lead:+.3f}s\n"
                f"v=({release_velocity[0]:+.2f}, {release_velocity[1]:+.2f}, "
                f"{release_velocity[2]:+.2f}) m/s  NN eta idx={eta_index}"
            )
            markers.append(release_text)

            impact_text = self._marker(8, "impact_text", Marker.TEXT_VIEW_FACING)
            impact_text.pose.position = self._point(landing + np.array([0, 0, 0.10]))
            impact_text.scale.z = 0.045
            self._color(impact_text, (0.75, 0.90, 1.0, 1.0))
            impact_text.text = (
                f"PREDICTED {'HIT' if evaluation.selected_hit else 'MISS'}  "
                f"error={selected.error_xy:.3f}m\n"
                f"impact=({landing[0]:+.3f}, {landing[1]:+.3f}, "
                f"{landing[2]:+.3f})m  flight={flight_time:.3f}s"
            )
            markers.append(impact_text)

            # Goal/bin acceptance footprint.
            goal = self._marker(9, "throw_goal", Marker.CYLINDER)
            goal.pose.position = self._point([
                self._goal_xy[0], self._goal_xy[1], self._impact_z + 0.005,
            ])
            goal.scale.x = goal.scale.y = 2.0 * self._goal_radius
            goal.scale.z = 0.01
            self._color(goal, (0.10, 0.95, 0.20, 0.28))
            markers.append(goal)

            # Every legal eta waypoint's predicted landing locus.
            locus = self._marker(10, "release_candidate_landings", Marker.POINTS)
            locus.scale.x = locus.scale.y = 0.028
            self._color(locus, (1.0, 0.75, 0.05, 0.90))
            locus.points = [
                self._point([
                    candidate.landing[0], candidate.landing[1],
                    self._impact_z + 0.012,
                ])
                for candidate in candidates
            ]
            markers.append(locus)

            best_marker = self._marker(11, "best_release_landing", Marker.SPHERE)
            best_marker.pose.position = self._point(
                best.landing + np.array([0.0, 0.0, 0.018])
            )
            best_marker.scale.x = best_marker.scale.y = best_marker.scale.z = 0.055
            self._color(
                best_marker,
                (0.15, 1.0, 0.20, 0.95) if evaluation.feasible_exists
                else (1.0, 0.55, 0.05, 0.95),
            )
            markers.append(best_marker)

            best_text = self._marker(12, "best_release_text", Marker.TEXT_VIEW_FACING)
            best_text.pose.position = self._point(
                best.landing + np.array([0.0, 0.0, 0.13])
            )
            best_text.scale.z = 0.042
            self._color(best_text, (1.0, 0.95, 0.70, 1.0))
            best_text.text = (
                f"best eta={best.eta:.3f} idx={best.index} "
                f"error={best.error_xy:.3f}m\n"
                f"feasible={'YES' if evaluation.feasible_exists else 'NO'} "
                f"recommended lead={recommended_lead:+.3f}s"
            )
            markers.append(best_text)

            self._publisher.publish(MarkerArray(markers=markers))
            self._node.get_logger().info(
                "Throw RViz preview: "
                f"release t={ts[release_index]:.3f}s idx={release_index} "
                f"p=({release_pos[0]:+.3f},{release_pos[1]:+.3f},{release_pos[2]:+.3f}) "
                f"v=({release_velocity[0]:+.3f},{release_velocity[1]:+.3f},"
                f"{release_velocity[2]:+.3f})m/s -> "
                f"impact=({landing[0]:+.3f},{landing[1]:+.3f},{landing[2]:+.3f}) "
                f"after {flight_time:.3f}s error={selected.error_xy:.3f}m"
            )
            return evaluation
        except Exception as exc:  # visualization must never stop robot control
            self._node.get_logger().warn(f"Throw RViz preview failed: {exc}")
            return None

    def log_post_throw(self, evaluation: ThrowEvaluation | None) -> None:
        """Print the model-based hit/miss and alternate-release audit."""
        if evaluation is None:
            self._node.get_logger().warn(
                "[throw-eval] unavailable: ballistic preview calculation failed"
            )
            return

        selected = evaluation.selected
        status = "PREDICTED HIT" if evaluation.selected_hit else "PREDICTED MISS"
        summary = (
            f"[throw-eval] {status}: selected idx={selected.index} "
            f"eta={selected.eta:.3f} t={selected.time:.3f}s "
            f"landing=({selected.landing[0]:+.3f},{selected.landing[1]:+.3f},"
            f"{selected.landing[2]:+.3f}) goal=({evaluation.goal_xy[0]:+.3f},"
            f"{evaluation.goal_xy[1]:+.3f}) error={selected.error_xy:.3f}m "
            f"radius={evaluation.goal_radius:.3f}m"
        )
        if evaluation.selected_hit:
            self._node.get_logger().info(summary)
            return

        self._node.get_logger().warn(summary)
        best = evaluation.best
        feasibility = "FOUND" if evaluation.feasible_exists else "NOT FOUND"
        self._node.get_logger().warn(
            f"[throw-eval] feasible release {feasibility}: "
            f"count={evaluation.feasible_count}, best idx={best.index} "
            f"eta={best.eta:.3f} t={best.time:.3f}s "
            f"landing=({best.landing[0]:+.3f},{best.landing[1]:+.3f},"
            f"{best.landing[2]:+.3f}) error={best.error_xy:.3f}m; "
            f"recommended release_lead="
            f"{evaluation.recommended_release_lead:+.3f}s "
            f"(current {evaluation.release_lead:+.3f}s)"
        )
