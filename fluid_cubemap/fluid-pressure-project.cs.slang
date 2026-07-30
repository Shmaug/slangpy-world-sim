import Math;
import SphericalField;

uniform SphericalVelocityField velocity;
uniform SphericalScalarField divergence;
uniform SphericalScalarField pressure_correction[2];
// mip level to read from for `resample`
uniform uint src_mip_level;
// mip level of pressure_correction (and divergence) that `step` relaxes this dispatch;
// `compute_divergence` and `apply` always operate at mip 0
uniform uint dst_mip_level;

Texture2D<float> surface_height;
Texture2D<float> surface_height2;
SamplerState linear_sampler;
uniform float terrain_height;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_divergence(uint3 thread: SV_DispatchThreadID)
{
    CubeCoord cell = CubeCoord(thread.xy, thread.z % 6, thread.z / 6, 0);
    if (!divergence.bounds_check(cell))
        return;

    CubeCoord x0 = velocity.get_neighbor(cell, int2(-1, 0));
    CubeCoord x1 = velocity.get_neighbor(cell, int2( 1, 0));
    CubeCoord y0 = velocity.get_neighbor(cell, int2( 0,-1));
    CubeCoord y1 = velocity.get_neighbor(cell, int2( 0, 1));
    CubeCoord z0, z1;
    bool z0_ok = vertical_neighbor(cell, -1, velocity.vertical_resolution, z0);
    bool z1_ok = vertical_neighbor(cell,  1, velocity.vertical_resolution, z1);

    float3 dir    = cube_uvw_to_dir(float3(float2(cell.texel + 0.5) / velocity.resolution, cell.face));
    float3 dir_x0 = cube_uvw_to_dir(float3(float2(x0.texel + 0.5) / velocity.resolution, x0.face));
    float3 dir_x1 = cube_uvw_to_dir(float3(float2(x1.texel + 0.5) / velocity.resolution, x1.face));
    float3 dir_y0 = cube_uvw_to_dir(float3(float2(y0.texel + 0.5) / velocity.resolution, y0.face));
    float3 dir_y1 = cube_uvw_to_dir(float3(float2(y1.texel + 0.5) / velocity.resolution, y1.face));

    let frame = CubeMapTangentFrame(dir);
    float3 vel_center = velocity.load(cell);
    float vel_x0 = frame.to_local(velocity.load_world_space(x0, dir)).x;
    float vel_x1 = frame.to_local(velocity.load_world_space(x1, dir)).x;
    float vel_y0 = frame.to_local(velocity.load_world_space(y0, dir)).y;
    float vel_y1 = frame.to_local(velocity.load_world_space(y1, dir)).y;
    float vel_z0 = z0_ok ? frame.to_local3(velocity.load_world_space(z0, dir)).z : -vel_center.z;
    float vel_z1 = z1_ok ? frame.to_local3(velocity.load_world_space(z1, dir)).z : 0;

    float dr = velocity.thickness / velocity.vertical_resolution;
    float r = 1.0 + dr * (cell.layer + 0.5);
    
    // non-uniform central difference: neighbor spacing varies with where
    // `cell` sits on its face, so the two half-differences are weighted by
    // the actual geodesic distance to each neighbor instead of assumed to
    // both be one texel wide
    float dist_x0 = sphere_geodesic_distance(dir, dir_x0) * r;
    float dist_x1 = sphere_geodesic_distance(dir, dir_x1) * r;
    float dist_y0 = sphere_geodesic_distance(dir, dir_y0) * r;
    float dist_y1 = sphere_geodesic_distance(dir, dir_y1) * r;

    // radial term: same (face,u,v) footprint at layer+-1, so no parallel
    // transport is needed (the outward normal is identical at every layer);
    // at the closed bottom the missing neighbor is mirrored with a sign
    // flip so the wall-normal velocity is exactly zero at the wall; at the
    // open top the ghost is vacuum (zero velocity, zero density) so mass
    // is free to flow out through the last layer
    float w_center = vel_center.z;
    float w_z0 = z0_ok ? velocity.load(z0).z : -w_center; // mirror at bottom
    float w_z1 = z1_ok ? velocity.load(z1).z : 0; // vacuum at open top

    // anelastic divergence constraint div(rho0 * u) = 0: rho0 only varies
    // radially, so it factors out of the horizontal terms as a plain scalar,
    // but the vertical term needs to be in flux form (rho0 sampled at each
    // neighbor layer) to capture d(rho0)/dr; at the closed bottom there's no
    // neighbor layer to evaluate rho0 at, so the ghost reuses this cell's own
    // rho0 (consistent with the w sign-flip ghost above), while the open top
    // ghost uses rho0=0 (vacuum), consistent with the w_z1 ghost above
    float rho0 = velocity.sample_density(r);
    float rho0_z0 = z0_ok ? velocity.sample_density(r - dr) : rho0;
    float rho0_z1 = z1_ok ? velocity.sample_density(r + dr) : 0;

    // both ghosts (bottom mirror, top vacuum) sit at the same canonical
    // spacing dr, so the flux derivative always spans 2*dr regardless of
    // which boundary (if any) is being ghosted
    divergence.store(cell, rho0 * (
        (vel_x1 - vel_x0) / (dist_x0 + dist_x1) +
        (vel_y1 - vel_y0) / (dist_y0 + dist_y1)) +
        (rho0_z1 * w_z1 - rho0_z0 * w_z0) / (2.0 * dr));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void step(uint3 thread: SV_DispatchThreadID)
{
    CubeCoord cell = CubeCoord(thread.xy, thread.z % 6, thread.z / 6, 0);
    if (!divergence.bounds_check(cell, dst_mip_level))
        return;

    float dr = velocity.thickness / velocity.vertical_resolution;
    float r = 1.0 + dr * (cell.layer + 0.5);

    CubeCoord x0 = divergence.get_neighbor(cell, int2(-1, 0), dst_mip_level);
    CubeCoord x1 = divergence.get_neighbor(cell, int2( 1, 0), dst_mip_level);
    CubeCoord y0 = divergence.get_neighbor(cell, int2( 0,-1), dst_mip_level);
    CubeCoord y1 = divergence.get_neighbor(cell, int2( 0, 1), dst_mip_level);

    CubeCoord z0, z1;
    bool z0_ok = vertical_neighbor(cell, -1, velocity.vertical_resolution, z0);
    bool z1_ok = vertical_neighbor(cell,  1, velocity.vertical_resolution, z1);

    // non-uniform Jacobi stencil for the Laplacian: on a uniform grid every
    // neighbor's weight collapses to 1 (matching the old p*0.25 average),
    // but here each axis's pair of neighbor distances can differ, so the
    // weights are derived from the standard unequal-spacing second-
    // derivative finite-difference formula instead of a plain average
    float3 dir = cube_uvw_to_dir(float3(float2(cell.texel + 0.5) / velocity.resolution, cell.face));
    uint mip_res = velocity.resolution >> dst_mip_level;
    float3 dir_xm = cube_uvw_to_dir(float3(float2(x0.texel + 0.5) / mip_res, x0.face));
    float3 dir_xp = cube_uvw_to_dir(float3(float2(x1.texel + 0.5) / mip_res, x1.face));
    float3 dir_ym = cube_uvw_to_dir(float3(float2(y0.texel + 0.5) / mip_res, y0.face));
    float3 dir_yp = cube_uvw_to_dir(float3(float2(y1.texel + 0.5) / mip_res, y1.face));
    float d_x0 = sphere_geodesic_distance(dir, dir_xm) * r;
    float d_x1 = sphere_geodesic_distance(dir, dir_xp) * r;
    float d_y0 = sphere_geodesic_distance(dir, dir_ym) * r;
    float d_y1 = sphere_geodesic_distance(dir, dir_yp) * r;
    // both ghosts (bottom Neumann copy, top Dirichlet 0) sit at the same
    // canonical spacing dr as a real neighbor would
    float d_z0 = dr;
    float d_z1 = dr;

    float d_sum_x = d_x0 + d_x1;
    float d_sum_y = d_y0 + d_y1;
    float d_sum_z = d_z0 + d_z1;

    float w_x0 = 2.0 / (d_x0 * d_sum_x);
    float w_x1 = 2.0 / (d_x1 * d_sum_x);
    float w_y0 = 2.0 / (d_y0 * d_sum_y);
    float w_y1 = 2.0 / (d_y1 * d_sum_y);
    float w_z0 = 2.0 / (d_z0 * d_sum_z);
    float w_z1 = 2.0 / (d_z1 * d_sum_z);

    // vertical (radial) neighbors: uniform spacing, independent of the
    // horizontal mip level. At the closed bottom there's no flux across the
    // wall, so a Neumann (zero-gradient) condition applies: the ghost value
    // is just a copy of this cell's own (current-iterate) pressure. At the
    // open top the atmosphere drains to ambient pressure 0 just past the
    // last layer (Dirichlet).
    float p_center = pressure_correction[0].load<1>(cell, dst_mip_level);

    float p = 0;
    p += w_x0 * pressure_correction[0].load<1>(x0, dst_mip_level).x;
    p += w_x1 * pressure_correction[0].load<1>(x1, dst_mip_level).x;
    p += w_y0 * pressure_correction[0].load<1>(y0, dst_mip_level).x;
    p += w_y1 * pressure_correction[0].load<1>(y1, dst_mip_level).x;
    p += w_z0 * (z0_ok ? pressure_correction[0].load<1>(z0, dst_mip_level).x : p_center);
    p += w_z1 * (z1_ok ? pressure_correction[0].load<1>(z1, dst_mip_level).x : 0);

    p -= divergence.load<1>(cell, dst_mip_level).x;

    // a cell fully walled in by terrain has every weight collapse to 0 --
    // fall back to holding its current value rather than dividing by 0
    float total_w = w_x0 + w_x1 + w_y0 + w_y1 + w_z0 + w_z1;
    pressure_correction[1].store(cell, total_w > 1e-8 ? p / total_w : p_center, dst_mip_level);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resample(uint3 thread: SV_DispatchThreadID)
{
    CubeCoord cell = CubeCoord(thread.xy, thread.z % 6, thread.z / 6, 0);
    if (!pressure_correction[0].bounds_check(cell, dst_mip_level))
        return;
    
    uint dst_res = pressure_correction[0].resolution >> dst_mip_level;

    float3 dir = cube_uvw_to_dir(float3((float2(cell.texel) + 0.5) / dst_res, cell.face));

    // pass the cell's own layer through -- omitting it would silently
    // default to layer 0, scrambling every other layer's multires solve
    pressure_correction[1].store(cell, pressure_correction[0].sample_layer<1>(dir, 0, cell.layer, src_mip_level), dst_mip_level);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void apply(uint3 thread: SV_DispatchThreadID)
{
    CubeCoord cell = CubeCoord(thread.xy, thread.z % 6, thread.z / 6, 0);
    if (!velocity.bounds_check(cell))
        return;

    CubeCoord x0 = pressure_correction[0].get_neighbor(cell, int2(-1, 0));
    CubeCoord x1 = pressure_correction[0].get_neighbor(cell, int2( 1, 0));
    CubeCoord y0 = pressure_correction[0].get_neighbor(cell, int2(0, -1));
    CubeCoord y1 = pressure_correction[0].get_neighbor(cell, int2(0, 1));
    CubeCoord z0, z1;
    bool z0_ok = vertical_neighbor(cell, -1, velocity.vertical_resolution, z0);
    bool z1_ok = vertical_neighbor(cell,  1, velocity.vertical_resolution, z1);

    // vertical pressure gradient: same Neumann (copy-of-center) bottom
    // ghost and Dirichlet (ambient 0) top ghost as the Jacobi step above
    float dr = velocity.thickness / velocity.vertical_resolution;
    float r = 1.0 + dr * (cell.layer + 0.5);

    float p_center = pressure_correction[0].load<1>(cell);
    float px0 = pressure_correction[0].load<1>(x0);
    float px1 = pressure_correction[0].load<1>(x1);
    float py0 = pressure_correction[0].load<1>(y0);
    float py1 = pressure_correction[0].load<1>(y1);
    float pz0 = z0_ok ? pressure_correction[0].load<1>(z0) : p_center;
    float pz1 = z1_ok ? pressure_correction[0].load<1>(z1) : 0;

    // same non-uniform central difference as compute_divergence, so the
    // gradient removed here matches the spacing the divergence/Laplacian
    // were computed with
    float3 dir    = cube_uvw_to_dir(float3(float2(cell.texel + 0.5) / velocity.resolution, cell.face));
    float3 dir_x0 = cube_uvw_to_dir(float3(float2(x0.texel + 0.5) / velocity.resolution, x0.face));
    float3 dir_x1 = cube_uvw_to_dir(float3(float2(x1.texel + 0.5) / velocity.resolution, x1.face));
    float3 dir_y0 = cube_uvw_to_dir(float3(float2(y0.texel + 0.5) / velocity.resolution, y0.face));
    float3 dir_y1 = cube_uvw_to_dir(float3(float2(y1.texel + 0.5) / velocity.resolution, y1.face));
    float dist_x0 = sphere_geodesic_distance(dir, dir_x0) * r;
    float dist_x1 = sphere_geodesic_distance(dir, dir_x1) * r;
    float dist_y0 = sphere_geodesic_distance(dir, dir_y0) * r;
    float dist_y1 = sphere_geodesic_distance(dir, dir_y1) * r;

    float2 grad = float2((px1 - px0) / (dist_x0 + dist_x1), (py1 - py0) / (dist_y0 + dist_y1));
    // both ghosts (bottom Neumann copy, top Dirichlet 0) sit at the same
    // canonical spacing dr, so this always spans 2*dr regardless of which
    // boundary (if any) is being ghosted
    float grad_z = (pz1 - pz0) / (2.0 * dr);

    // the correction was solved for as u -= (dt/rho0) * grad(p) folded into
    // a single constant-coefficient Poisson solve (see compute_divergence),
    // so rho0 has to be divided back out here rather than in `step`
    float rho0 = velocity.sample_density(r);

    float3 vel = velocity.load(cell);
    vel.xy -= grad / rho0;
    vel.z -= grad_z / rho0;

    velocity.store(cell, vel);
}