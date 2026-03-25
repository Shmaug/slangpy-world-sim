// CO-FLIP (Covector FLIP) fluid advection
//
// Instead of advecting velocity components as independent scalars (standard FLIP/semi-Lagrangian),
// CO-FLIP advects the impulse field as a covector using the pullback of the flow map.
// This preserves circulation (Kelvin's theorem) and produces sharper vortical structures.
//
// The impulse m relates to velocity u by: u = P(m) where P is the Leray-Helmholtz
// (divergence-free) projection. After covector advection writes the transported impulse
// to the velocity grid, the existing pressure projection step recovers divergence-free velocity.
//
// Entry points:
//   advect_coflip_grid     - Grid-based semi-Lagrangian covector advection
//   advect_coflip_particle - Particle-based forward covector advection
//   coflip_particle_to_grid - Transfer particle impulse back to MAC grid

import Math;
import FluidGrid;
import ParticleMap;

uniform FluidGrid grid;
uniform ParticleMap particle_map;
uniform float advect_dt;

// RK4 integration of a position through the velocity field
float2 integrate_velocity(float2 grid_pos, float dt)
{
    float2 k1 = grid.sample_velocity(grid_pos);
    float2 k2 = grid.sample_velocity(grid_pos + k1 * dt / 2.0);
    float2 k3 = grid.sample_velocity(grid_pos + k2 * dt / 2.0);
    float2 k4 = grid.sample_velocity(grid_pos + k3 * dt);
    grid_pos += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    grid_pos = (grid_pos + grid.resolution) % grid.resolution;
    return grid_pos;
}

// Shortest wrapped difference on a periodic domain
float2 wrap_diff(float2 a, float2 b, float2 res)
{
    float2 d = a - b;
    d -= round(d / res) * res;
    return d;
}

// Covector pullback: m_new = F^T * m, where F is the backward-map Jacobian
float2 covector_pullback(float2x2 F, float2 m)
{
    // mul(row_vec, matrix) = F^T * col_vec in standard linear algebra notation
    return mul(m, F);
}

// Compute backward-map Jacobian ∂ψ/∂x at `pos` using finite differences of RK4 backtracing.
// Returns the 2x2 Jacobian and writes the backtraced origin to `out_origin`.
float2x2 backward_map_jacobian(float2 pos, float dt, out float2 origin)
{
    static const float eps = 0.5;
    float2 res = float2(grid.resolution);

    origin           = integrate_velocity(pos,                  dt);
    float2 origin_px = integrate_velocity(pos + float2(eps, 0), dt);
    float2 origin_py = integrate_velocity(pos + float2(0, eps), dt);

    float2 dpsi_dx = wrap_diff(origin_px, origin, res) / eps;
    float2 dpsi_dy = wrap_diff(origin_py, origin, res) / eps;

    return float2x2(dpsi_dx.x, dpsi_dy.x,
                    dpsi_dx.y, dpsi_dy.y);
}

// ============================================================
// Grid-based CO-FLIP: semi-Lagrangian covector advection
//
// For each MAC face, backtrace to find the origin, compute the
// Jacobian of the backward flow map, and apply the covector
// pullback to the sampled impulse.
// ============================================================
[shader("compute")]
[numthreads(8, 8, 1)]
void advect_coflip_grid(uint3 thread : SV_DispatchThreadID)
{
    uint2 cell = thread.xy;

    // --- x-face impulse at (cell.x, cell.y + 0.5) ---
    if (cell.x < grid.resolution.x + 1 && cell.y < grid.resolution.y)
    {
        float2 origin;
        float2x2 F = backward_map_jacobian(float2(cell.x, cell.y + 0.5), -advect_dt, origin);
        float2 m = grid.sample_velocity(origin);
        grid.velocity_x_rw[cell] = covector_pullback(F, m).x;
    }

    // --- y-face impulse at (cell.x + 0.5, cell.y) ---
    if (cell.x < grid.resolution.x && cell.y < grid.resolution.y + 1)
    {
        float2 origin;
        float2x2 F = backward_map_jacobian(float2(cell.x + 0.5, cell.y), -advect_dt, origin);
        float2 m = grid.sample_velocity(origin);
        grid.velocity_y_rw[cell] = covector_pullback(F, m).y;
    }

    // --- pressure at cell center (scalar, standard semi-Lagrangian) ---
    if (cell.x < grid.resolution.x && cell.y < grid.resolution.y)
    {
        float2 pos = float2(cell.x + 0.5, cell.y + 0.5);
        float2 origin = integrate_velocity(pos, -advect_dt);
        grid.pressure_rw[cell] = grid.sample_pressure(origin);
    }
}

// ============================================================
// Particle-based CO-FLIP: forward advection with covector transport
//
// Spawns one particle per cell, advects it forward, computes the
// forward-map Jacobian J, and applies J^{-T} to the impulse.
// ============================================================
[shader("compute")]
[numthreads(8, 8, 1)]
void advect_coflip_particle(uint3 thread : SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution))
        return;

    float2 pos = float2(cell) + 0.5;
    static const float eps = 0.5;
    float2 res = float2(grid.resolution);

    float2 m = grid.sample_velocity(pos);
    float pressure = grid.sample_pressure(pos);

    // Forward-advect the particle and two neighbors for finite-difference Jacobian
    float2 new_pos    = integrate_velocity(pos,                  advect_dt);
    float2 new_pos_px = integrate_velocity(pos + float2(eps, 0), advect_dt);
    float2 new_pos_py = integrate_velocity(pos + float2(0, eps), advect_dt);

    // Forward-map Jacobian J = ∂φ/∂x
    float2 dphi_dx = wrap_diff(new_pos_px, new_pos, res) / eps;
    float2 dphi_dy = wrap_diff(new_pos_py, new_pos, res) / eps;

    // Covector transport under forward map: m' = J^{-T} * m
    //   J = [a b; c d]  =>  J^{-T} = (1/det) [d -c; -b a]
    float det = dphi_dx.x * dphi_dy.y - dphi_dy.x * dphi_dx.y;
    float2 m_transported;
    if (abs(det) > 1e-6)
    {
        float inv_det = 1.0 / det;
        m_transported = inv_det * float2(
             dphi_dy.y * m.x - dphi_dx.y * m.y,
            -dphi_dy.x * m.x + dphi_dx.x * m.y
        );
    }
    else
    {
        m_transported = m; // degenerate Jacobian: fall back to untransformed impulse
    }

    Particle p;
    p.pos_in_cell = frac(new_pos);
    p.velocity = m_transported;
    p.pressure = pressure;
    particle_map.insert(int(new_pos.y) * grid.resolution.x + int(new_pos.x), p);
}

// ============================================================
// Particle-to-grid transfer (bilinear splat to MAC grid)
//
// Identical to vanilla FLIP — the covector transport has already
// been applied on the particles; this step just scatters values
// back onto the staggered grid.
// ============================================================
[shader("compute")]
[numthreads(8, 8, 1)]
void coflip_particle_to_grid(uint3 thread: SV_DispatchThreadID)
{
    int2 dst_cell = thread.xy;
    if (any(dst_cell >= grid.resolution + 1))
        return;

    float vel_x = 0;
    float vel_y = 0;
    float pressure = 0;

    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            int2 cell = (dst_cell + int2(x, y) + grid.resolution) % grid.resolution;
            uint2 range = particle_map.get_range(cell.y * grid.resolution.x + cell.x);
            for (uint i = range.x; i < range.y; i++)
            {
                Particle p = particle_map.get(i);

                // Unwrapped position for correct distance at grid boundaries
                float2 pos = float2(dst_cell + int2(x, y)) + p.pos_in_cell;

                float2 dx = abs(pos - float2(dst_cell.x, dst_cell.y + 0.5));
                float wx = max(0.0, 1.0 - dx.x) * max(0.0, 1.0 - dx.y);
                vel_x += p.velocity.x * wx;

                float2 dy = abs(pos - float2(dst_cell.x + 0.5, dst_cell.y));
                float wy = max(0.0, 1.0 - dy.x) * max(0.0, 1.0 - dy.y);
                vel_y += p.velocity.y * wy;

                float2 dc = abs(pos - float2(dst_cell.x + 0.5, dst_cell.y + 0.5));
                float wc = max(0.0, 1.0 - dc.x) * max(0.0, 1.0 - dc.y);
                pressure += p.pressure * wc;
            }
        }
    }

    if (dst_cell.x < grid.resolution.x + 1 && dst_cell.y < grid.resolution.y)
        grid.velocity_x_rw[dst_cell] = vel_x;
    if (dst_cell.x < grid.resolution.x && dst_cell.y < grid.resolution.y + 1)
        grid.velocity_y_rw[dst_cell] = vel_y;
    if (dst_cell.x < grid.resolution.x && dst_cell.y < grid.resolution.y)
        grid.pressure_rw[dst_cell]   = pressure;
}
