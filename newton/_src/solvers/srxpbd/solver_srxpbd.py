from newton._src.sim import model
import warp as wp
import numpy as np

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .kernels import (
    apply_particle_deltas,
    solve_particle_shape_contacts,
    solve_shape_matching,
)


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
        self.particle_q_rest = model.particle_q

    def apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
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

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                particle_qd, 
                model.particle_flags,
                model.particle_mass,
                particle_deltas,
                dt,
                model.particle_max_velocity,
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
                                # outputs
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                        )
                        particle_q, particle_qd = self.apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

            if model.particle_count and model.particle_group_count > 0:

                # Pre-compute which groups have mass (do this once, not every iteration)
                # NOTE: This could take a while if there's many spheres/groups, potentially leverage ParticleFlags 
                # to speed this up
                if not hasattr(self, '_group_has_mass_cache'):
                    self._group_has_mass_cache = []
                    for group_id in range(model.particle_group_count):
                        group_particle_indices = model.particle_groups[group_id]
                        group_masses = model.particle_mass.numpy()[group_particle_indices.numpy()]
                        has_mass = bool(np.any(group_masses > 0.0))
                        self._group_has_mass_cache.append(has_mass)

                # Process each particle group separately
                for group_id in range(model.particle_group_count):
                    # Skip static groups
                    if not self._group_has_mass_cache[group_id]:
                        continue

                    # Get particle indices for this group
                    group_particle_indices = model.particle_groups[group_id]
                    group_particle_count = len(group_particle_indices.numpy())

                    # Create local delta for this group
                    local_delta = wp.zeros_like(particle_deltas)

                    wp.launch(
                        kernel=solve_shape_matching,
                        dim=1,
                        inputs=[
                            particle_q, 
                            self.particle_q_rest,
                            model.particle_mass,
                            group_particle_indices,
                            group_particle_count,
                            local_delta,
                        ],
                        outputs=[particle_deltas],
                        device=model.device
                    )

                # Apply all accumulated deltas at once
                particle_q, particle_qd = self.apply_particle_deltas(
                    model, state_in, state_out, particle_deltas, dt
                )

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            return state_out
