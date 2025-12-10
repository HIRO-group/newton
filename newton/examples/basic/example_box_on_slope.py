import numpy as np
import warp as wp
import newton
import newton.examples

MU = 0.0

class Slope:
    center = wp.vec3(0.0, -0.5, 0.1)
    rot = wp.quat(-0.258819, 0, 0, 0.9659258)  # -30 degrees around x-axis
    hx, hy, hz = 0.4, 0.8, 0.1

    def add_slope(builder):
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = MU
        builder.add_shape_box(
            -1,
            wp.transform(
                Slope.center,
                Slope.rot,
            ),
            hx=Slope.hx,
            hy=Slope.hy,
            hz=Slope.hz,
            cfg=cfg,
        )
        return builder

class Box:
    mass = 4.0
    hx, hy, hz = 0.1, 0.1, 0.1
    up_slope_offset = -0.7

    def pose_on_slope():
        local_z = wp.vec3(0.0, 0.0, 1.0)
        n = wp.quat_rotate(Slope.rot, local_z)
        base_center = Slope.center + n * (Slope.hz + Box.hz)
        local_y = wp.vec3(0.0, 1.0, 0.0)
        tangent = wp.quat_rotate(Slope.rot, local_y)
        center = base_center + tangent * Box.up_slope_offset
        return center, Slope.rot
    
    def add_sphere_packed_box(builder, num_spheres):
        center, rot = Box.pose_on_slope()
        dim_x = dim_y = dim_z = num_spheres
        radius_mean = Box.hx / num_spheres
        cell_x = (2 * Box.hx - 2 * radius_mean) / (dim_x - 1)
        cell_y = (2 * Box.hy - 2 * radius_mean) / (dim_y - 1)
        cell_z = (2 * Box.hz - 2 * radius_mean) / (dim_z - 1)
        total_mass = Box.mass
        num_particles = dim_x * dim_y * dim_z
        mass_per_particle = total_mass / num_particles
        pos_corner = center + wp.quat_rotate(rot, wp.vec3(-Box.hx + radius_mean, -Box.hy + radius_mean, -Box.hz + radius_mean))
        builder.add_particle_grid(
            pos=wp.vec3(pos_corner[0], pos_corner[1], pos_corner[2]),
            rot=rot,
            vel=wp.vec3(0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_x,
            cell_y=cell_y,
            cell_z=cell_z,
            mass=mass_per_particle,
            jitter=0.0,
            radius_mean=radius_mean,
            radius_std=0.0,
        )
        return builder


class Example:
    def __init__(self, viewer, args):
        self.sim_time = 0.0
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.num_spheres = args.num_spheres

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder = Slope.add_slope(builder)
        builder = Box.add_sphere_packed_box(builder=builder, num_spheres=self.num_spheres)

        self.model = builder.finalize()

        self.model.particle_mu = MU
        self.model.soft_contact_mu = MU
        self.solver = newton.solvers.SolverSRXPBD(self.model,
                                                    iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)


    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
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
        "-n",
        "--num_spheres",
        type=int,
        default=5,
        help="Number of spheres per box dimension for SRXPBD experiment",
    )
    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = Example(viewer, args=args)
    newton.examples.run(example, args)
