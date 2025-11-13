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
# Example Basic Shapes
#
# Shows how to programmatically creates a variety of
# collision shapes using the newton.ModelBuilder() API.
#
# Command: python -m newton.examples basic_shapes
#
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # z height to drop shapes from
        drop_z = 2.0
        
        
        solver_params = {
                "add_springs": True,
                "spring_ke": 1.0e3,
                "spring_kd": 1.0e1,
        }
        
        common_params = {
            "pos": wp.vec3(0.0, 0.0, 4.0),
            "rot": wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            "vel": wp.vec3(0.0, 0.0, 0.0),
            "dim_x": 64,
            "dim_y": 32,
            "cell_x": 0.1,
            "cell_y": 0.1,
            "mass": 0.1,
            "fix_left": True,
            "edge_ke": 1.0e1,
            "edge_kd": 0.0,
            "particle_radius": 0.05,
        }


        # builder.add_cloth_grid(**common_params, **solver_params)

        N = 4
        particles_per_cell = 3
        voxel_size = 0.5
        particle_spacing = voxel_size / particles_per_cell
        friction = 0.6

        builder.add_particle_grid(
            pos=wp.vec3(0.5 * particle_spacing),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=N * particles_per_cell,
            dim_y=N * particles_per_cell,
            dim_z=N * particles_per_cell,
            cell_x=particle_spacing,
            cell_y=particle_spacing,
            cell_z=particle_spacing,
            mass=1.0,
            jitter=0.0,
        )

        self.model = builder.finalize()

        # TODO
        self.model.particle_ke = 1.0e15
        self.model.particle_mu = friction
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0


        self.solver = newton.solvers.SolverSRXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for maximal-coordinate solvers like XPBD
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test(self):
        self.sphere_pos[2] = 0.5
        sphere_q = wp.transform(self.sphere_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "sphere at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, sphere_q, atol=1e-4),
            [0],
        )
        self.capsule_pos[2] = 1.0
        capsule_q = wp.transform(self.capsule_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "capsule at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, capsule_q, atol=1e-4),
            [1],
        )
        self.cylinder_pos[2] = 0.6
        cylinder_q = wp.transform(self.cylinder_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cylinder at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, cylinder_q, atol=1e-4),
            [2],
        )
        self.box_pos[2] = 0.25
        box_q = wp.transform(self.box_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "box at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, box_q, atol=0.1),
            [3],
        )
        # we only test that the bunny didn't fall through the ground and didn't slide too far
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "bunny at rest pose",
            lambda q, qd: q[2] > 0.01 and abs(q[0]) < 0.1 and abs(q[1] - 4.0) < 0.1,
            [4],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()
    viewer.show_particles = True

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example, args)
