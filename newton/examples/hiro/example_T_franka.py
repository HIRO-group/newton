# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example IK Franka (positions + rotations)
#
# Inverse kinematics on a Franka FR3 arm targeting the TCP (fr3_hand_tcp).
# - Single IKPositionObjective + IKRotationObjective
# - Gizmo controls the TCP target (with ViewerGL.log_gizmo)
#
# Command: python -m newton.examples ik_franka
###########################################################################

import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils


class Example:
    def __init__(self, viewer):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer

        # ------------------------------------------------------------------
        # Build a single FR3 (fixed base) + ground
        # ------------------------------------------------------------------
        self.scene = newton.ModelBuilder()

        franka = newton.ModelBuilder()
        franka.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )

        self.scene.add_builder(franka)
        self.scene.add_ground_plane()

        # add a table
        self.scene.add_shape_box(
            -1,
            wp.transform(
                wp.vec3(0.5, 0, 0.1),
                wp.quat_identity(),
            ),
            hx=0.4,
            hy=0.4,
            hz=0.1,
        )

        # Put the soft grid on top of the box
        # Box top is at z = 0.1 + 0.1 = 0.2
        # Soft grid is 2x2x2 cells of 0.1, so its height is 0.2, center should be at 0.2 + 0.1 = 0.3
        self.scene.add_soft_grid(
            pos=wp.vec3(0.5, 0, 0.2),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=1, dim_y=1, dim_z=1,
            cell_x=0.1, cell_y=0.1, cell_z=0.1,
            density=1000.0,
            k_mu=1e6,      # Very high stiffness
            k_lambda=1e6,  # Very high stiffness
            k_damp=1e3,    # High damping
            tri_ke=1e6,    # High triangle stiffness
            tri_ka=1e6,    # High area preservation
            tri_kd=1e3,    # High damping
        )

        self.graph = None
        self.model = self.scene.finalize()
        self.viewer.set_model(self.model)

        # states
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # ------------------------------------------------------------------
        # End effector
        # ------------------------------------------------------------------
        self.ee_index = 10  # hardcoded for now

        # Persistent gizmo transform (pass-by-ref mutated by viewer)
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])

        # ------------------------------------------------------------------
        # IK setup (single problem, single EE)
        # ------------------------------------------------------------------
        # residual layout:
        # [0..2]  : position (3)
        # [3..5]  : rotation (3)
        # [6..]   : joint limits (joint_coord_count)
        total_residuals = 6 + self.model.joint_coord_count

        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # Position objective
        self.pos_obj = ik.IKPositionObjective(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ee_tf)], dtype=wp.vec3),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=0,
        )

        # Rotation objective
        self.rot_obj = ik.IKRotationObjective(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.ee_tf))], dtype=wp.vec4),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=3,
        )

        # Joint limit objective
        self.obj_joint_limits = ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=6,
            weight=10.0,
        )

        # Variables the solver will update
        self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))

        self.ik_iters = 24
        self.franka_solver = ik.IKSolver(
            model=self.model,
            joint_q=self.joint_q,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.ANALYTIC,
        )

        self.capture()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.franka_solver.solve(iterations=self.ik_iters)

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transform and push into IK objectives."""
        self.pos_obj.set_target_position(0, wp.transform_get_translation(self.ee_tf))
        q = wp.transform_get_rotation(self.ee_tf)
        self.rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        self._push_targets_from_gizmos()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Register gizmo (viewer will draw & mutate transform in-place)
        self.viewer.log_gizmo("target_tcp", self.ee_tf)

        # Visualize the current articulated state
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.viewer.log_state(self.state)

        self.viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example, args)
