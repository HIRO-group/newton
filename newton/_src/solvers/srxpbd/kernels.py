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

import warp as wp

from ...core import velocity_at_point
from ...geometry import ParticleFlags
from ...sim import JointType
from ...utils import (
    vec_abs,
    vec_leaky_max,
    vec_leaky_min,
    vec_max,
    vec_min,
)


@wp.kernel
def apply_particle_shape_restitution(
    particle_x_new: wp.array(dtype=wp.vec3),
    particle_v_new: wp.array(dtype=wp.vec3),
    particle_x_old: wp.array(dtype=wp.vec3),
    particle_v_old: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    particle_ka: float,
    restitution: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    dt: float,
    relaxation: float,
    particle_v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    # x_new = particle_x_new[particle_index]
    v_new = particle_v_new[particle_index]
    px = particle_x_old[particle_index]
    v_old = particle_v_old[particle_index]

    X_wb = wp.transform_identity()
    # X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        # X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    # r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    rel_vel_old = wp.dot(n, v_old)
    rel_vel_new = wp.dot(n, v_new)

    if rel_vel_old < 0.0:
        # dv = -n * wp.max(-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0), 0.0)
        dv = n * (-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0))

        # compute inverse masses
        # w1 = particle_invmass[particle_index]
        # w2 = 0.0
        # if body_index >= 0:
        #     angular = wp.cross(r, n)
        #     q = wp.transform_get_rotation(X_wb)
        #     rot_angular = wp.quat_rotate_inv(q, angular)
        #     I_inv = body_I_inv[body_index]
        #     w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
        # denom = w1 + w2
        # if denom == 0.0:
        #     return

        wp.atomic_add(particle_v_out, tid, dv)


@wp.kernel
def compute_shape_matching_goals(
    particle_q: wp.array(dtype=wp.vec3),
    particle_q_init: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_count: int,
    goal_positions: wp.array(dtype=wp.vec3)
):
    tot_w = float(0.0)
    t = wp.vec3(0.0)
    t0 = wp.vec3(0.0)
    for i in range(particle_count):
        w = particle_mass[i]
        x = particle_q[i]
        x0 = particle_q_init[i]

        tot_w += w
        t += w * x
        t0 += w * x0

    t *= (1.0/ tot_w)
    t0 *= (1.0/ tot_w)

    # covariance A
    A = wp.mat33(0.0)
    for i in range(particle_count):
        w = particle_mass[i]
        x = particle_q[i]
        x0 = particle_q_init[i]
        pi = x - t
        qi = x0 - t0
        A += wp.outer(pi, qi) * w

    # polar decomposition via SVD
    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(A, U, S, V)
    R = U @ wp.transpose(V)

    if (wp.determinant(R) < 0.0): # TODO
        U[:,2] = -U[:,2]
        R = U @ wp.transpose(V) 
    
    for i in range(particle_count):
        x0 = particle_q_init[i]
        x = particle_q[i]
        goal_positions[i] = R * (x0 - t0) + t


@wp.kernel
def solve_shape_matching_constraints(
    particle_q: wp.array(dtype=wp.vec3),
    goal_positions: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_count: int,
    lambdas: wp.array(dtype=wp.vec3),
    compliance: float,
    dt: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid >= particle_count:
        return

    alpha_tilde = compliance / (dt * dt)
    w = particle_inv_mass[tid]
    
    # Constraint: C(x) = x - goal
    curr_x = particle_q[tid]
    goal = goal_positions[tid]
    C = curr_x - goal

    dx =  (goal-curr_x)
    wp.atomic_add(delta, tid, dx)

    # XPBD Lambda Update
    # lambda_old = lambdas[tid]
    # denom = w + alpha_tilde
    # d_lambda = (-C - alpha_tilde * lambda_old) / denom
    # lambdas[tid] = lambda_old + d_lambda
    # dx = w * d_lambda
    # wp.atomic_add(delta, tid, dx)


@wp.kernel
def solve_particle_shape_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    shape_material_mu: wp.array(dtype=float),
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    dt: float,
    relaxation: float,
    # outputs
    delta: wp.array(dtype=wp.vec3),
    body_delta: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    # take average material properties of shape and particle parameters
    mu = 0.5 * (particle_mu + shape_material_mu[shape_index])

    # body velocity
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]

    body_w = wp.spatial_bottom(body_v_s)
    body_v = wp.spatial_top(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])

    # relative velocity
    v = pv - bv

    # normal
    lambda_n = c
    delta_n = n * lambda_n

    # friction
    vn = wp.dot(n, v)
    vt = v - n * vn

    # compute inverse masses
    w1 = particle_invmass[particle_index]
    w2 = 0.0
    if body_index >= 0:
        angular = wp.cross(r, n)
        q = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q, angular)
        I_inv = body_I_inv[body_index]
        w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
    denom = w1 + w2
    if denom == 0.0:
        return

    lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
    delta_f = wp.normalize(vt) * lambda_f
    delta_total = (delta_f - delta_n) / denom * relaxation

    wp.atomic_add(delta, particle_index, w1 * delta_total)

    if body_index >= 0:
        delta_t = wp.cross(r, delta_total)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_total, delta_t))


@wp.kernel
def solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    # outputs
    deltas: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # hash grid has not been built yet
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    w1 = particle_invmass[i]

    # particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    delta = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) != 0 and index != i:
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]

            # compute inverse masses
            w2 = particle_invmass[index]
            denom = w1 + w2

            if err <= k_cohesion and denom > 0.0:
                n = n / d
                vrel = v - particle_v[index]

                # normal
                lambda_n = err
                delta_n = n * lambda_n

                # friction
                vn = wp.dot(n, vrel)
                vt = v - n * vn

                lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                delta_f = wp.normalize(vt) * lambda_f
                delta += (delta_f - delta_n) / denom

    wp.atomic_add(deltas, i, delta * w1 * relaxation)


@wp.kernel
def solve_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    dt: float,
    lambdas: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)

    if l == 0.0:
        return

    n = xij / l

    c = l - rest
    grad_c_xi = n
    grad_c_xj = -1.0 * n

    wi = invmass[i]
    wj = invmass[j]

    denom = wi + wj

    # Note strict inequality for damping -- 0 damping is ok
    if denom <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_c_dot_v = dt * wp.dot(grad_c_xi, vij)  # Note: dt because from the paper we want x_i - x^n, not v...
    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_c_dot_v) / ((1.0 + gamma) * denom + alpha)

    dxi = wi * dlambda * grad_c_xi
    dxj = wj * dlambda * grad_c_xj

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, dxi)
    wp.atomic_add(delta, j, dxj)


@wp.kernel
def bending_constraint(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    indices: wp.array2d(dtype=int),
    rest: wp.array(dtype=float),
    bending_properties: wp.array2d(dtype=float),
    dt: float,
    lambdas: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    eps = 1.0e-6

    ke = bending_properties[tid, 0]
    kd = bending_properties[tid, 1]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    if i == -1 or j == -1 or k == -1 or l == -1:
        return

    rest_angle = rest[tid]

    x1 = x[i]
    x2 = x[j]
    x3 = x[k]
    x4 = x[l]

    v1 = v[i]
    v2 = v[j]
    v3 = v[k]
    v4 = v[l]

    w1 = invmass[i]
    w2 = invmass[j]
    w3 = invmass[k]
    w4 = invmass[l]

    n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1
    n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2
    e = x4 - x3

    n1_length = wp.length(n1)
    n2_length = wp.length(n2)
    e_length = wp.length(e)

    # Check for degenerate cases
    if n1_length < eps or n2_length < eps or e_length < eps:
        return

    n1_hat = n1 / n1_length
    n2_hat = n2 / n2_length
    e_hat = e / e_length

    cos_theta = wp.dot(n1_hat, n2_hat)
    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    c = theta - rest_angle

    grad_x1 = -n1_hat * e_length
    grad_x2 = -n2_hat * e_length
    grad_x3 = -n1_hat * wp.dot(x1 - x4, e_hat) - n2_hat * wp.dot(x2 - x4, e_hat)
    grad_x4 = -n1_hat * wp.dot(x3 - x1, e_hat) - n2_hat * wp.dot(x3 - x2, e_hat)

    denominator = (
        w1 * wp.length_sq(grad_x1)
        + w2 * wp.length_sq(grad_x2)
        + w3 * wp.length_sq(grad_x3)
        + w4 * wp.length_sq(grad_x4)
    )

    # Note strict inequality for damping -- 0 damping is ok
    if denominator <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_dot_v = dt * (wp.dot(grad_x1, v1) + wp.dot(grad_x2, v2) + wp.dot(grad_x3, v3) + wp.dot(grad_x4, v4))

    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_dot_v) / ((1.0 + gamma) * denominator + alpha)

    delta0 = w1 * dlambda * grad_x1
    delta1 = w2 * dlambda * grad_x2
    delta2 = w3 * dlambda * grad_x3
    delta3 = w4 * dlambda * grad_x4

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, delta0)
    wp.atomic_add(delta, j, delta1)
    wp.atomic_add(delta, k, delta2)
    wp.atomic_add(delta, l, delta3)


@wp.kernel
def solve_tetrahedra(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    indices: wp.array(dtype=int, ndim=2),
    rest_matrix: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    # act = activation[tid]

    # k_mu = materials[tid, 0]
    # k_lambda = materials[tid, 1]
    # k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    # v0 = v[i]
    # v1 = v[j]
    # v2 = v[k]
    # v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = rest_matrix[tid]
    inv_QT = wp.transpose(Dm)

    inv_rest_volume = wp.determinant(Dm) * 6.0

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)

    C = float(0.0)
    dC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    compliance = float(0.0)

    stretching_compliance = relaxation
    volume_compliance = relaxation

    num_terms = 2
    for term in range(0, num_terms):
        if term == 0:
            # deviatoric, stable
            C = tr - 3.0
            dC = F * 2.0
            compliance = stretching_compliance
        elif term == 1:
            # volume conservation
            C = wp.determinant(F) - 1.0
            dC = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))
            compliance = volume_compliance

        if C != 0.0:
            dP = dC * inv_QT
            grad1 = wp.vec3(dP[0][0], dP[1][0], dP[2][0])
            grad2 = wp.vec3(dP[0][1], dP[1][1], dP[2][1])
            grad3 = wp.vec3(dP[0][2], dP[1][2], dP[2][2])
            grad0 = -grad1 - grad2 - grad3

            w = (
                wp.dot(grad0, grad0) * w0
                + wp.dot(grad1, grad1) * w1
                + wp.dot(grad2, grad2) * w2
                + wp.dot(grad3, grad3) * w3
            )

            if w > 0.0:
                alpha = compliance / dt / dt
                if inv_rest_volume > 0.0:
                    alpha *= inv_rest_volume
                dlambda = -C / (w + alpha)

                wp.atomic_add(delta, i, w0 * dlambda * grad0)
                wp.atomic_add(delta, j, w1 * dlambda * grad1)
                wp.atomic_add(delta, k, w2 * dlambda * grad2)
                wp.atomic_add(delta, l, w3 * dlambda * grad3)
                # wp.atomic_add(particle.num_corr, id0, 1)
                # wp.atomic_add(particle.num_corr, id1, 1)
                # wp.atomic_add(particle.num_corr, id2, 1)
                # wp.atomic_add(particle.num_corr, id3, 1)

    # C_Spherical
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # r_s_inv = 1.0/r_s
    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv
    # alpha = 1.0

    # C_D
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0
    # alpha = 1.0

    # grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    # grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    # grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    # grad0 = (grad1 + grad2 + grad3) * (0.0 - 1.0)

    # denom = (
    #     wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    # )
    # multiplier = C / (denom + 1.0 / (k_mu * dt * dt * rest_volume))

    # delta0 = grad0 * multiplier
    # delta1 = grad1 * multiplier
    # delta2 = grad2 * multiplier
    # delta3 = grad3 * multiplier

    # # hydrostatic part
    # J = wp.determinant(F)

    # C_vol = J - alpha
    # # dCdx = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))*wp.transpose(Dm)

    # # grad1 = wp.vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # # grad2 = wp.vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # # grad3 = wp.vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    # s = inv_rest_volume / 6.0
    # grad1 = wp.cross(x20, x30) * s
    # grad2 = wp.cross(x30, x10) * s
    # grad3 = wp.cross(x10, x20) * s
    # grad0 = -(grad1 + grad2 + grad3)

    # denom = (
    #     wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    # )
    # multiplier = C_vol / (denom + 1.0 / (k_lambda * dt * dt * rest_volume))

    # delta0 += grad0 * multiplier
    # delta1 += grad1 * multiplier
    # delta2 += grad2 * multiplier
    # delta3 += grad3 * multiplier

    # # # apply forces
    # # wp.atomic_sub(delta, i, delta0 * w0 * relaxation)
    # # wp.atomic_sub(delta, j, delta1 * w1 * relaxation)
    # # wp.atomic_sub(delta, k, delta2 * w2 * relaxation)
    # # wp.atomic_sub(delta, l, delta3 * w3 * relaxation)


@wp.kernel
def solve_tetrahedra2(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    indices: wp.array(dtype=int, ndim=2),
    pose: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    # act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    # k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    # v0 = v[i]
    # v1 = v[j]
    # v2 = v[k]
    # v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = pose[tid]

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # C_sqrt
    # tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)
    # r_s = wp.sqrt(abs(tr - 3.0))
    # C = r_s

    # if (r_s == 0.0):
    #     return

    # if (tr < 3.0):
    #     r_s = 0.0 - r_s

    # dCdx = F*wp.transpose(Dm)*(1.0/r_s)
    # alpha = 1.0 + k_mu / k_lambda

    # C_Neo
    r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    if r_s == 0.0:
        return
    # tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)
    # if (tr < 3.0):
    #     r_s = -r_s
    r_s_inv = 1.0 / r_s
    C = r_s
    dCdx = F * wp.transpose(Dm) * r_s_inv
    alpha = 1.0 + k_mu / k_lambda

    # C_Spherical
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # r_s_inv = 1.0/r_s
    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv
    # alpha = 1.0

    # C_D
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0
    # alpha = 1.0

    grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    grad0 = (grad1 + grad2 + grad3) * (0.0 - 1.0)

    denom = (
        wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    )
    multiplier = C / (denom + 1.0 / (k_mu * dt * dt * rest_volume))

    delta0 = grad0 * multiplier
    delta1 = grad1 * multiplier
    delta2 = grad2 * multiplier
    delta3 = grad3 * multiplier

    # hydrostatic part
    J = wp.determinant(F)

    C_vol = J - alpha
    # dCdx = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))*wp.transpose(Dm)

    # grad1 = wp.vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = wp.vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = wp.vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    s = inv_rest_volume / 6.0
    grad1 = wp.cross(x20, x30) * s
    grad2 = wp.cross(x30, x10) * s
    grad3 = wp.cross(x10, x20) * s
    grad0 = -(grad1 + grad2 + grad3)

    denom = (
        wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    )
    multiplier = C_vol / (denom + 1.0 / (k_lambda * dt * dt * rest_volume))

    delta0 += grad0 * multiplier
    delta1 += grad1 * multiplier
    delta2 += grad2 * multiplier
    delta3 += grad3 * multiplier

    # apply forces
    wp.atomic_sub(delta, i, delta0 * w0 * relaxation)
    wp.atomic_sub(delta, j, delta1 * w1 * relaxation)
    wp.atomic_sub(delta, k, delta2 * w2 * relaxation)
    wp.atomic_sub(delta, l, delta3 * w3 * relaxation)


@wp.kernel
def apply_particle_deltas(
    x_orig: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    delta: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    x0 = x_orig[tid]
    xp = x_pred[tid]

    # constraint deltas
    d = delta[tid]

    x_new = xp + d
    v_new = (x_new - x0) / dt

    # enforce velocity limit to prevent instability
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    x_out[tid] = x_new
    v_out[tid] = v_new
