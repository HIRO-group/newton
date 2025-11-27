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
import warp as wp
import newton
import newton.examples


class Box:
    pos = wp.vec3(0.0, 2.0, 1.0)
    rot = wp.quat(0.1830127, 0.1830127, 0.6830127, 0.6830127)

    @staticmethod
    def get_tr_urdf(builder):
        urdf_path = 'newton/examples/assets/hiro/urdfs/pink_box/pink_box.urdf'
        builder.add_urdf(urdf_path, xform=wp.transform(p=Box.pos, q=Box.rot),
                        floating=True, ignore_inertial_definitions=True,
                        scale=1.0)
        return builder

    @staticmethod
    def get_MorphIt_spheres():
        pass

    @staticmethod
    def get_default_spheres():
        pass

class Example:
    def __init__(self, viewer, experiment):
        self.sim_time = 0.0
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        if experiment in ["gt", "b1", "b3"]:
            builder = Box.get_tr_urdf(builder)
        if experiment == "b2":
            builder = Box.get_default_spheres(builder)
        if experiment == "ours":
            builder = Box.get_MorphIt_spheres(builder)

        self.model = builder.finalize()

        if experiment == "gt":
            self.solver = newton.solvers.SolverMuJoCo(self.model, iterations=100, ls_iterations=50, njmax=100)
        elif experiment == "b1":
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
        elif experiment == "b2":
            self.solver = newton.solvers.SolverSRXPBD(self.model, iterations=10)
        elif experiment == "b3":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
        elif experiment == "ours":
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

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--experiment",
        help="Type of experiment",
        type=str,
        choices=["gt", "b1", "b2", "b3", "ours"],
        default="gt",
    )
    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = Example(viewer, experiment=args.experiment)
    newton.examples.run(example, args)
