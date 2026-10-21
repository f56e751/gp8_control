"""Small real MuJoCo scenes test contact, distance and cloned-FK semantics."""
from types import SimpleNamespace as NS
import unittest

import mujoco
import numpy as np

from gp8_control.rl.safe_v2.monitor import RouteContactMonitor, StaticCommandGuard


XML = """<mujoco><option timestep="0.002" gravity="0 0 0"/>
<worldbody>
 <geom name="floor" type="plane" size="2 2 .01"/>
 <geom name="gp8_belt_surface" type="box" pos="2 0 0" size=".3 .5 .005"/>
 <body name="yaskawa_robot">
  <geom name="fixed_base" type="box" pos="-2 0 .05" size=".05 .05 .05"/>
  <body name="link6" pos="0 0 .1">
   <joint name="S_axis" type="slide" axis="1 0 0"/>
   <joint name="L_axis" type="slide" axis="0 1 0"/>
   <joint name="U_axis" type="slide" axis="0 0 1"/>
   <joint name="R_axis" type="hinge" axis="1 0 0"/>
   <joint name="B_axis" type="hinge" axis="0 1 0"/>
   <joint name="T_axis" type="hinge" axis="0 0 1"/>
   <geom name="pusher_collision" type="sphere" size=".02" mass="1"/>
  </body>
 </body>
 <body name="box1" pos=".05 0 .1"><freejoint name="box1_free"/>
  <geom name="box1_geom" type="sphere" size=".02" mass=".2"/>
 </body>
 <body name="box2" pos="1 0 .1"><freejoint name="box2_free"/>
  <geom name="box2_geom" type="sphere" size=".02" mass=".2"/>
 </body>
</worldbody></mujoco>"""


class _Core:
    _FREE = 'free'

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self._dt = float(self.model.opt.timestep)
        self._base_p, self._base_R = np.zeros(3), np.eye(3)
        self._belt_top_w = 0.
        self._box_geom_ids = [int(self.model.geom(f'box{i}_geom').id) for i in (1, 2)]
        self._box_body_ids = [int(self.model.body(f'box{i}').id) for i in (1, 2)]
        self._box_state, self._box_serial = ['on_belt'] * 2, [1, 2]
        self._box_manipulated = [False, False]
        self._box_joints = [self.data.joint(f'box{i}_free') for i in (1, 2)]
        self._grabbed, self._vacuum_on = None, False
        self._link6_id = int(self.model.body('link6').id)
        self.cfg = NS(despawn_y=-.8)
        self.safe_residual_context = None
        self.safe_runner = NS(_active_target=None, _chain_action=None, _status='IDLE',
            _skill_name_for_object=lambda target: 'throw', cfg=NS(MAX_REACH=.65))
        mujoco.mj_forward(self.model, self.data)
        self.safe_monitor = RouteContactMonitor(self)

    def sim_time(self):
        return float(self.data.time)

    def _to_base(self, position):
        return np.asarray(position)

    def step(self, count=1):
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
            self.safe_monitor.observe()

    def move_object(self, index, xyz):
        self._box_joints[index].qpos[:3] = xyz
        self._box_joints[index].qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)

    def push_context(self, prepare_end):
        self.safe_runner._active_target = NS(sim_object_id=1)
        self.safe_residual_context = dict(task='push', phase='push_route', target_id=1,
            start_sim_time=0., prepare_end=prepare_end, stroke_end=1., duration=2.)


class RouteMonitorTests(unittest.TestCase):
    def test_surface_distance_integrates_in_physical_seconds(self):
        core = _Core()
        core.step(20)
        summary = core.safe_monitor.snapshot()
        self.assertAlmostEqual(summary['current_clearance_m'], .01, places=7)
        self.assertAlmostEqual(summary['minimum_clearance_m'], .01, places=7)
        self.assertAlmostEqual(summary['proximity_integral'], (2/3)**2 * .04, places=7)
        self.assertEqual(summary['contact_events'], 0)
        integral = summary['proximity_integral']
        core.safe_monitor.observe()
        self.assertEqual(core.safe_monitor.snapshot()['proximity_integral'], integral)

    def test_normal_grasp_exemption_applies_to_dense_distance(self):
        core = _Core()
        core.safe_runner._active_target = NS(sim_object_id=1)
        core.safe_runner._status = 'POSITIONING'
        core._vacuum_on = True
        core.step(5)
        self.assertEqual(core.safe_monitor.snapshot()['proximity_integral'], 0.)
        core._vacuum_on = False
        core.step()
        self.assertGreater(core.safe_monitor.snapshot()['proximity_integral'], 0.)

    def test_held_object_surface_also_detects_neighbor_risk(self):
        core = _Core()
        core.data.qpos[0] = -1.
        core.move_object(0, [0., 0., .1])
        core.move_object(1, [.05, 0., .1])
        core._grabbed, core._box_manipulated[0] = 0, True
        core.safe_runner._active_target = NS(sim_object_id=1)
        core.step()
        self.assertAlmostEqual(core.safe_monitor.snapshot()['current_clearance_m'], .01, places=7)
        self.assertGreater(core.safe_monitor.snapshot()['proximity_integral'], 0.)

    def test_actual_floor_contact_is_environment_risk(self):
        core = _Core()
        core.data.qpos[2] = -.09
        core.move_object(0, [1., 1., .1])
        core.step()
        summary = core.safe_monitor.snapshot()
        self.assertGreater(summary['environment_contact_events'], 0)
        self.assertGreater(summary['environment_impulse_ns'], 0.)
        self.assertEqual(summary['object_contact_events'], 0)
        base = int(core.model.geom('fixed_base').id)
        floor = int(core.model.geom('floor').id)
        self.assertIsNone(core.safe_monitor.classify_pair(base, floor))

    def test_tool_belt_support_only_exempt_during_actual_push_stroke(self):
        for prepare_end, expected_normal in ((0., True), (1., False)):
            with self.subTest(prepare_end=prepare_end):
                core = _Core()
                core.data.qpos[0], core.data.qpos[2] = 2., -.08
                core.move_object(0, [1., 1., .1])
                core.push_context(prepare_end)
                core.step()
                values = core.safe_monitor.snapshot()
                self.assertEqual(values['environment_contact_events'] == 0, expected_normal)
                if expected_normal:
                    self.assertGreater(values['normal_contact_samples'], 0)


class CommandGuardTests(unittest.TestCase):
    def setUp(self):
        self.core = _Core()
        self.guard = StaticCommandGuard(self.core)
        self.grid = np.arange(0., .104, .004)
        self.nominal = np.zeros((len(self.grid), 6))
        self.window = dict(edit_start=0., edit_end=.1)

    def test_candidate_floor_penetration_rejected_without_live_mutation(self):
        candidate = self.nominal.copy()
        candidate[:, 2] = -.09
        state = self.core.data.qpos.copy()
        result = self.guard.check(candidate, self.nominal, self.grid, self.window)
        self.assertFalse(result['accepted'])
        self.assertLess(result['min_tool_bottom_m'], 0.)
        np.testing.assert_array_equal(self.core.data.qpos, state)
        self.assertEqual(self.core.sim_time(), 0.)

    def test_existing_native_penetration_only_permitted_if_not_worsened(self):
        native = self.nominal.copy()
        native[:, 2] = -.09
        self.assertTrue(self.guard.check(native, native, self.grid, self.window)['accepted'])
        worse = native.copy()
        worse[:, 2] -= .01
        self.assertFalse(self.guard.check(worse, native, self.grid, self.window)['accepted'])
        better = native.copy()
        better[:, 2] += .01
        self.assertTrue(self.guard.check(better, native, self.grid, self.window)['accepted'])

    def test_clear_lift_and_protected_prefix_are_allowed(self):
        candidate = self.nominal.copy()
        candidate[:, 2] = .1
        self.assertTrue(self.guard.check(candidate, self.nominal, self.grid, self.window)['accepted'])
        # An earlier decision's prefix is outside the current edit authority.
        candidate[:5, 2] = -.09
        protected = np.zeros(len(self.grid), bool)
        protected[:5] = True
        result = self.guard.check(candidate, self.nominal, self.grid,
                                  dict(self.window, protected_mask=protected))
        self.assertTrue(result['accepted'])


if __name__ == '__main__':
    unittest.main()
