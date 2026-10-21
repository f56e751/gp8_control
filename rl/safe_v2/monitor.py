"""Native contact telemetry, early clearance shaping and static command checks.

All solver measurements are read-only. Geometry previews use a separate
MjData, preserve the live model/data, and are a sampled command guard rather
than a guarantee about dynamically tracked motion.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from gp8_control.backends.mujoco_sim import MJ_JOINTS
from gp8_control.rl.safe_contact_monitor import SafeContactMonitor


def _movable_geoms(core, robot_geoms):
    model = core.model
    root = int(model.body('yaskawa_robot').id)
    root_weld = int(model.body_weldid[root])
    return {g for g in robot_geoms if int(model.body_weldid[int(model.geom_bodyid[g])]) != root_weld}


def _collidable(model, gid):
    return bool(int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))


def _environment_geoms(core, robot_geoms):
    boxes = set(map(int, core._box_geom_ids))
    return {g for g in range(core.model.ngeom)
            if g not in boxes and g not in robot_geoms and _collidable(core.model, g)}


def _near_enough(model, data, geom1, geom2, cap):
    """Bounding spheres (and plane support) reject distant narrow-phase calls."""
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    if int(model.geom_type[geom2]) == plane:
        normal = np.asarray(data.geom_xmat[geom2]).reshape(3, 3)[:, 2]
        height = float(np.dot(data.geom_xpos[geom1] - data.geom_xpos[geom2], normal))
        return height - float(model.geom_rbound[geom1]) <= cap
    if int(model.geom_type[geom1]) == plane:
        return _near_enough(model, data, geom2, geom1, cap)
    lower = float(np.linalg.norm(data.geom_xpos[geom1] - data.geom_xpos[geom2]))
    lower -= float(model.geom_rbound[geom1]) + float(model.geom_rbound[geom2])
    return lower <= cap


def _distance(model, data, geom1, geom2, cap):
    if not _near_enough(model, data, geom1, geom2, cap):
        return float(cap)
    value = float(mujoco.mj_geomDistance(model, data, geom1, geom2, float(cap), None))
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite MuJoCo geometry distance')
    return min(value, float(cap))


class RouteContactMonitor(SafeContactMonitor):
    """v1 contact definition plus environment events and early object clearance.

    Only movable robot geometry is used for proximity/static contacts. Permanent
    base mounting contacts are exempt. Tool/belt support is exempt exclusively
    during the native protected Push stroke, never during approach or transfer.
    Proximity is the minimum surface distance to an object, not a TCP distance;
    its capped squared deficit is integrated in actual physical seconds.
    """

    def __init__(self, core, *, proximity_period=.02, safe_clearance=.03,
                 clearance_cap=.15, **kwargs):
        super().__init__(core, **kwargs)
        if not (0 < proximity_period and 0 < safe_clearance <= clearance_cap
                and np.isfinite([proximity_period, safe_clearance, clearance_cap]).all()):
            raise ValueError('Clearance and sampling parameters must be finite and positive')
        self._movable_geoms = _movable_geoms(core, self._robot_geoms)
        self._proximity_geoms = {g for g in self._movable_geoms if _collidable(core.model, g)}
        self._environment_geoms = _environment_geoms(core, self._robot_geoms)
        self._belt_geoms = {g for g in self._environment_geoms
                            if 'belt' in str(core.model.geom(g).name or '').lower()}
        self.proximity_period, self.safe_clearance = float(proximity_period), float(safe_clearance)
        self.clearance_cap = float(clearance_cap)
        self.proximity_integral = 0.
        self.minimum_clearance_m = self.current_clearance_m = self.clearance_cap
        self._proximity_rate = 0.
        self._last_proximity_sample = -float('inf')
        self._last_phase_signature = None
        self._last_proximity_observe = None

    def classify_pair(self, geom1, geom2, context=None):
        classified = super().classify_pair(geom1, geom2, context)
        if classified is not None:
            return classified
        context = self.phase_context() if context is None else context
        geom1, geom2 = int(geom1), int(geom2)
        if geom2 in self._movable_geoms and geom1 in self._environment_geoms:
            geom1, geom2 = geom2, geom1
        if geom1 not in self._movable_geoms or geom2 not in self._environment_geoms:
            return None
        if (geom1 in self._tool_geoms and geom2 in self._belt_geoms
                and context['phase'] == 'push_stroke'):
            return 'normal_target', (), ()
        return 'robot_environment', (), ()

    def _sample_clearance(self, context):
        core, minimum = self.core, self.clearance_cap
        live = [(i, int(core._box_geom_ids[i])) for i, state in enumerate(core._box_state)
                if state != core._FREE]
        for robot_gid in self._proximity_geoms:
            for _, object_gid in live:
                classified = self.classify_pair(robot_gid, object_gid, context)
                if classified is not None and classified[0] == 'normal_target':
                    continue
                minimum = min(minimum, _distance(core.model, core.data,
                                                robot_gid, object_gid, self.clearance_cap))
        held = getattr(core, '_grabbed', None)
        if held is not None and 0 <= int(held) < len(core._box_state):
            held_gid = int(core._box_geom_ids[int(held)])
            for index, object_gid in live:
                if index != int(held):
                    minimum = min(minimum, _distance(core.model, core.data,
                                                    held_gid, object_gid, self.clearance_cap))
        self.current_clearance_m = float(minimum)
        self.minimum_clearance_m = min(self.minimum_clearance_m, self.current_clearance_m)
        deficit = np.clip((self.safe_clearance - minimum) / self.safe_clearance, 0., 1.)
        self._proximity_rate = float(deficit**2)

    def observe(self):
        now = float(self.core.sim_time())
        if self._last_proximity_observe is not None and now <= self._last_proximity_observe + 1e-12:
            return
        super().observe()
        context = self.phase_context()
        signature = (context['phase'], context['target_id'],
                     context.get('next_grasp_target_id'), getattr(self.core, '_grabbed', None))
        if (now - self._last_proximity_sample >= self.proximity_period - 1e-9
                or signature != self._last_phase_signature):
            self._sample_clearance(context)
            self._last_proximity_sample, self._last_phase_signature = now, signature
        self.proximity_integral += self._proximity_rate * float(self.core._dt)
        self._last_proximity_observe = now

    def snapshot(self):
        values = super().snapshot()
        environment = [event for event in self.events if event['kind'] == 'robot_environment']
        environment_impulse = float(sum(event['impulse'] for event in environment))
        values.update(proximity_integral=float(self.proximity_integral),
                      minimum_clearance_m=float(self.minimum_clearance_m),
                      current_clearance_m=float(self.current_clearance_m),
                      clearance_cap_m=self.clearance_cap,
                      environment_contact_events=len(environment),
                      object_contact_events=values['contact_events'] - len(environment),
                      environment_impulse_ns=environment_impulse,
                      object_impulse_ns=float(values['impulse']) - environment_impulse)
        return values


def _tool_bottom(model, data, gids, base_z):
    minimum = float('inf')
    for gid in gids:
        rotation = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
        size, kind = model.geom_size[gid], int(model.geom_type[gid])
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            extent = float(np.dot(np.abs(rotation[2]), size))
        elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            extent = float(size[0])
        elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            extent = float(size[0] + abs(rotation[2, 2]) * size[1])
        elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            extent = float(size[0] * np.linalg.norm(rotation[2, :2]) + abs(rotation[2, 2]) * size[1])
        else:
            extent = float(model.geom_rbound[gid])
        minimum = min(minimum, float(data.geom_xpos[gid, 2]) - extent - base_z)
    return None if not math.isfinite(minimum) else minimum


class StaticCommandGuard:
    """Sparse exact-geometry checks on cloned FK, with native non-worsening.

    The command must keep at least 2 mm static clearance where native motion
    already provides it. Existing native contact/penetration is permitted only
    when not worsened (0.5 mm numerical tolerance). Dynamic object avoidance is
    learned from measured proximity; this guard concerns static environment.
    """

    def __init__(self, core, *, sample_period=.02, maximum_samples=64,
                 clearance=.002, tolerance=.0005):
        self.core, self.preview = core, mujoco.MjData(core.model)
        helper = core.safe_monitor if isinstance(getattr(core, 'safe_monitor', None), RouteContactMonitor) else RouteContactMonitor(core)
        self.robot_geoms = {g for g in helper._movable_geoms if _collidable(core.model, g)}
        self.environment_geoms = helper._environment_geoms
        self.tool_geoms = self.robot_geoms & helper._tool_geoms
        self.addresses = [int(core.model.jnt_qposadr[int(core.model.joint(name).id)]) for name in MJ_JOINTS]
        self.sample_period, self.maximum_samples = float(sample_period), int(maximum_samples)
        self.clearance, self.tolerance = float(clearance), float(tolerance)
        if (not np.isfinite([sample_period, clearance, tolerance]).all()
                or sample_period <= 0 or maximum_samples < 2 or clearance < 0 or tolerance < 0):
            raise ValueError('Invalid geometry preview parameters')

    def _held_transform(self):
        core, model = self.core, self.core.model
        held = getattr(core, '_grabbed', None)
        if held is None:
            return None
        held = int(held)
        if not 0 <= held < len(core._box_state) or core._box_state[held] == core._FREE:
            return None
        object_bid = int(model.geom_bodyid[int(core._box_geom_ids[held])])
        joint = int(model.body_jntadr[object_bid])
        if joint < 0 or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            return None
        link = int(getattr(core, '_link6_id', model.jnt_bodyid[int(model.joint(MJ_JOINTS[-1]).id)]))
        self.preview.qpos[:] = core.data.qpos
        mujoco.mj_kinematics(model, self.preview)
        link_r = self.preview.xmat[link].reshape(3, 3).copy()
        position = link_r.T @ (self.preview.xpos[object_bid] - self.preview.xpos[link])
        rotation = link_r.T @ self.preview.xmat[object_bid].reshape(3, 3)
        return dict(gid=int(core._box_geom_ids[held]), link=link,
                    address=int(model.jnt_qposadr[joint]), position=position, rotation=rotation)

    def _forward(self, joints, held):
        core, data, model = self.core, self.preview, self.core.model
        data.qpos[:] = core.data.qpos
        data.qpos[self.addresses] = joints
        mujoco.mj_kinematics(model, data)
        if held is not None:
            rotation = data.xmat[held['link']].reshape(3, 3)
            position = data.xpos[held['link']] + rotation @ held['position']
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray((rotation @ held['rotation']).ravel()))
            address = held['address']
            data.qpos[address:address+3], data.qpos[address+3:address+7] = position, quaternion
            mujoco.mj_kinematics(model, data)

    def check(self, candidate, nominal, grid, window):
        candidate, nominal, grid = np.asarray(candidate, float), np.asarray(nominal, float), np.asarray(grid, float)
        if (candidate.ndim != 2 or candidate.shape[1:] != (6,) or candidate.shape != nominal.shape
                or grid.shape != (len(candidate),) or not np.isfinite(candidate).all()
                or not np.isfinite(nominal).all() or not np.isfinite(grid).all()
                or np.any(np.diff(grid) <= 0)):
            raise ValueError('Expected finite matching Nx6 commands and an increasing time grid')
        active = np.any(np.abs(candidate - nominal) > 1e-11, axis=1)
        active &= (grid >= float(window.get('edit_start', grid[0])) - 1e-9)
        active &= (grid <= float(window.get('edit_end', grid[-1])) + 1e-9)
        protected = window.get('protected_mask')
        if protected is not None:
            protected = np.asarray(protected, bool)
            if protected.shape != active.shape:
                raise ValueError('Protected sample mask has the wrong shape')
            active &= ~protected
        indices = np.flatnonzero(active)
        if not len(indices):
            return dict(accepted=True, reason='identity_or_protected', checked_samples=0,
                        min_tool_bottom_m=None, min_clearance_m=None)
        step = max(1, int(round(self.sample_period / float(np.median(np.diff(grid))))))
        indices = np.unique(np.r_[indices[::step], indices[-1]])
        if len(indices) > self.maximum_samples:
            indices = indices[np.unique(np.linspace(0, len(indices)-1, self.maximum_samples).astype(int))]
        held = self._held_transform()
        context = getattr(self.core, 'safe_residual_context', None) or {}
        release = window.get('release_time', context.get('release_time'))
        min_clearance, min_bottom = .05, float('inf')
        model, pairs_checked = self.core.model, 0
        for index in indices:
            hold_here = held if release is None or grid[index] <= float(release) + 1e-9 else None
            moving = set(self.robot_geoms)
            if hold_here is not None:
                moving.add(hold_here['gid'])
            self._forward(nominal[index], hold_here)
            native = {(g, e): _distance(model, self.preview, g, e, .05)
                      for g in moving for e in self.environment_geoms}
            self._forward(candidate[index], hold_here)
            bottom = _tool_bottom(model, self.preview, self.tool_geoms, float(self.core._base_p[2]))
            if bottom is not None:
                min_bottom = min(min_bottom, bottom)
            for (geom, environment), baseline in native.items():
                distance = _distance(model, self.preview, geom, environment, .05)
                pairs_checked += 1
                min_clearance = min(min_clearance, distance)
                if distance < min(self.clearance, baseline) - self.tolerance:
                    return dict(accepted=False, reason='new_or_worsened_static_contact',
                        checked_samples=int(np.searchsorted(indices, index)) + 1,
                        pairs_checked=pairs_checked, failed_time=float(grid[index]),
                        failed_geoms=[int(geom), int(environment)],
                        nominal_clearance_m=float(baseline), candidate_clearance_m=float(distance),
                        min_tool_bottom_m=None if not math.isfinite(min_bottom) else min_bottom,
                        min_clearance_m=float(min_clearance))
        return dict(accepted=True, reason='static_geometry_nonworsening', checked_samples=len(indices),
                    pairs_checked=pairs_checked, min_tool_bottom_m=None if not math.isfinite(min_bottom) else min_bottom,
                    min_clearance_m=float(min_clearance))


def command_static_guard(runner, candidate_samples, nominal_samples, grid, window):
    """Five-argument adapter callback; cache one private FK preview per core."""
    guard = getattr(runner.core, '_safe_v2_static_guard', None)
    if guard is None:
        guard = StaticCommandGuard(runner.core)
        runner.core._safe_v2_static_guard = guard
    return guard.check(candidate_samples, nominal_samples, grid, window)
