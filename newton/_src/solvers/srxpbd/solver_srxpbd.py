import warp as wp
import numpy as np

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .kernels import (
    apply_particle_deltas,
    apply_deltas_particles_for_shape_matching,
    solve_particle_shape_contacts,
    solve_shape_matching,
    # remove_momentum_from_shape_matching_d
)


def compute_angular_momentum(particle_q, particle_qd, particle_m):
    """
    """
    M = particle_m.sum()
    com = np.sum(particle_q * particle_m[:, None], axis=0) / M
    vcom = np.sum(particle_qd * particle_m[:, None], axis=0) / M

    r = particle_q - com[None, :]
    vrel = particle_qd - vcom[None, :]
    angular_momentum = np.sum(particle_m[:, None] * np.cross(r, vrel), axis=0)
    return wp.vec3(float(angular_momentum[0]), float(angular_momentum[1]), float(angular_momentum[2]))


def compute_linear_momentum(particle_qd, particle_m):
    """
    Compute total linear momentum of particles.
    """
    linear_momentum = np.sum(particle_m[:, None] * particle_qd, axis=0)
    return wp.vec3(float(linear_momentum[0]), float(linear_momentum[1]), float(linear_momentum[2]))


class SolverSRXPBD(SolverBase):
    """
    Similar to SolverXPBD. Only includes contact handling + shape matching constraints.
    This solver assumes complete rigid bodies. No soft bodies.
    Rigid bodies are modeled as a collection of particles.
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 2,
        soft_body_relaxation: float = 0.9,
        soft_contact_relaxation: float = 0.9,
        rigid_contact_relaxation: float = 0.8,
        rigid_contact_con_weighting: bool = True,
        enable_restitution: bool = False,
    ):
        super().__init__(model=model)
        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation
        self.soft_contact_relaxation = soft_contact_relaxation
        self.rigid_contact_relaxation = rigid_contact_relaxation
        self.rigid_contact_con_weighting = rigid_contact_con_weighting
        self.enable_restitution = enable_restitution

        # helper variables to track constraint resolution vars
        self._particle_delta_counter = 0
        self.particle_q_rest = wp.clone(
            model.particle_q)   # clone instead of reference

    def apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
        shape_matching: bool = False,
    ):
        if state_in.requires_grad:
            particle_q = state_out.particle_q
            particle_qd = state_out.particle_qd
            # allocate new particle arrays so gradients can be tracked correctly without overwriting
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                particle_qd = state_in.particle_qd
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        if not shape_matching:
            wp.launch(
                kernel=apply_particle_deltas,
                dim=model.particle_count,
                inputs=[
                    self.particle_q_init,
                    particle_q,
                    particle_qd,
                    model.particle_flags,
                    particle_deltas,
                    dt,
                    model.particle_max_velocity,
                ],
                outputs=[new_particle_q, new_particle_qd],
                device=model.device,
            )
        elif shape_matching:
            wp.launch(
                kernel=apply_deltas_particles_for_shape_matching,
                dim=1,
                inputs=[
                    particle_q,
                    particle_qd,
                    model.particle_flags,
                    particle_deltas,
                    model.particle_mass,
                    model.particle_count,
                    self.linear_momentum,
                    self.angular_momentum,
                    dt,
                ],
                outputs=[new_particle_q, new_particle_qd],
                device=model.device,

            )

        if state_in.requires_grad:
            state_out.particle_q = new_particle_q
            state_out.particle_qd = new_particle_qd

        return new_particle_q, new_particle_qd

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0

        model = self.model

        particle_q = None
        particle_qd = None
        particle_deltas = None
        body_deltas = None

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            if model.particle_count:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd
                self.particle_q_init = wp.clone(state_in.particle_q)
                self.particle_qd_init = wp.clone(state_in.particle_qd)
                particle_deltas = wp.empty_like(state_out.particle_qd)
                self.integrate_particles(model, state_in, state_out, dt)

            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                body_deltas = wp.empty_like(state_out.body_qd)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if model.particle_count:
                        if requires_grad and i > 0:
                            particle_deltas = wp.zeros_like(particle_deltas)
                        else:
                            particle_deltas.zero_()
                        if model.shape_count:
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    state_out.body_q,
                                    state_out.body_qd,
                                    model.body_com,
                                    model.body_inv_mass,
                                    model.body_inv_inertia,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_particle,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas, body_deltas],
                                device=model.device
                            )
                            particle_q, particle_qd = self.apply_particle_deltas(
                                model, state_in, state_out, particle_deltas, dt, shape_matching=False)

            if model.particle_count:
                self.angular_momentum = compute_angular_momentum(
                    particle_q=state_out.particle_q.numpy(),
                    particle_qd=state_out.particle_qd.numpy(),
                    particle_m=model.particle_mass.numpy())
                self.linear_momentum = compute_linear_momentum(
                    particle_qd=state_out.particle_qd.numpy(),
                    particle_m=model.particle_mass.numpy())

                particle_deltas.zero_()
                wp.launch(
                    kernel=solve_shape_matching,
                    dim=1,
                    inputs=[
                        particle_q,
                        self.particle_q_rest,
                        model.particle_mass,
                        model.particle_count,
                    ],
                    outputs=[particle_deltas],
                    device=model.device
                )

                particle_q, particle_qd = self.apply_particle_deltas(
                    model, state_in, state_out, particle_deltas, dt, shape_matching=True
                )

            # ang_momentum_after = compute_angular_momentum(
            #     particle_q=particle_q.numpy(),
            #     particle_qd=particle_qd.numpy(),
            #     particle_m=model.particle_mass.numpy(),
            # )
            # linear_momentum_after = compute_linear_momentum(
            #     particle_qd=particle_qd.numpy(),
            #     particle_m=model.particle_mass.numpy(),
            # )
            # L0 = np.array(self.angular_momentum, dtype=np.float32)
            # dL = np.array(ang_momentum_after, dtype=np.float32) - L0
            # rel = np.linalg.norm(dL) / (np.linalg.norm(L0) + 1e-30)
            # P0 = np.array(self.linear_momentum, dtype=np.float32)
            # dP = np.array(linear_momentum_after, dtype=np.float32) - P0
            # rel_P = np.linalg.norm(dP) / (np.linalg.norm(P0) + 1e-30)
            # print("||dL||:", np.linalg.norm(dL), " ||dP||:", np.linalg.norm(dP))
            # print("relative:", rel, " relative_P:", rel_P)
            # print("---------------------------------------------------")

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            return state_out
