import Math;
import FluidCube;

uniform FluidCube fluid;
// mip level of pressure_correction (and divergence) that `step` relaxes this dispatch;
// `compute_divergence` and `apply` always operate at mip 0
uniform uint mip_level;

StructuredBuffer<float>   divergence;
RWStructuredBuffer<float> divergence_rw;
StructuredBuffer<float>   pressure_correction;
RWStructuredBuffer<float> pressure_correction_rw;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_divergence(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= fluid.resolution) || thread.z >= 6)
        return;

    uint3 cell = thread;

    uint3 xm = fluid.get_neighbor(cell, int2(-1, 0));
    uint3 xp = fluid.get_neighbor(cell, int2( 1, 0));
    uint3 ym = fluid.get_neighbor(cell, int2( 0,-1));
    uint3 yp = fluid.get_neighbor(cell, int2( 0, 1));

    let frame = CubeMapTangentFrame(cube_uvw_to_dir(float3(float2(cell.xy + 0.5) / fluid.resolution, cell.z)));
    float vel_xm = frame.to_local(fluid.load_world_space_velocity_rw(xm)).x;
    float vel_xp = frame.to_local(fluid.load_world_space_velocity_rw(xp)).x;
    float vel_ym = frame.to_local(fluid.load_world_space_velocity_rw(ym)).y;
    float vel_yp = frame.to_local(fluid.load_world_space_velocity_rw(yp)).y;

    // non-uniform central difference: neighbor spacing varies with where
    // `cell` sits on its face, so the two half-differences are weighted by
    // the actual geodesic distance to each neighbor instead of assumed to
    // both be one texel wide
    float dist_xm = fluid.neighbor_distance(cell, xm);
    float dist_xp = fluid.neighbor_distance(cell, xp);
    float dist_ym = fluid.neighbor_distance(cell, ym);
    float dist_yp = fluid.neighbor_distance(cell, yp);

    divergence_rw[flatten_cube_texel(cell, fluid.resolution)] = (vel_xp - vel_xm) / (dist_xm + dist_xp) + (vel_yp - vel_ym) / (dist_ym + dist_yp);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void step(uint3 thread: SV_DispatchThreadID)
{
    uint res = fluid.resolution >> mip_level;
    if (any(thread.xy >= res) || thread.z >= 6)
        return;

    uint3 cell = thread;

    uint3 xm = fluid.get_neighbor(cell, int2(-1, 0), mip_level);
    uint3 xp = fluid.get_neighbor(cell, int2( 1, 0), mip_level);
    uint3 ym = fluid.get_neighbor(cell, int2( 0,-1), mip_level);
    uint3 yp = fluid.get_neighbor(cell, int2( 0, 1), mip_level);

    // non-uniform Jacobi stencil for the Laplacian: on a uniform grid every
    // neighbor's weight collapses to 1 (matching the old p*0.25 average),
    // but here each axis's pair of neighbor distances can differ, so the
    // weights are derived from the standard unequal-spacing second-
    // derivative finite-difference formula instead of a plain average
    float hxm = fluid.neighbor_distance(cell, xm, mip_level);
    float hxp = fluid.neighbor_distance(cell, xp, mip_level);
    float hym = fluid.neighbor_distance(cell, ym, mip_level);
    float hyp = fluid.neighbor_distance(cell, yp, mip_level);

    float wxm = 2.0 / (hxm * (hxm + hxp));
    float wxp = 2.0 / (hxp * (hxm + hxp));
    float wym = 2.0 / (hym * (hym + hyp));
    float wyp = 2.0 / (hyp * (hym + hyp));

    float p = 0;
    p += wxm * pressure_correction[fluid.flatten_index(xm, mip_level)];
    p += wxp * pressure_correction[fluid.flatten_index(xp, mip_level)];
    p += wym * pressure_correction[fluid.flatten_index(ym, mip_level)];
    p += wyp * pressure_correction[fluid.flatten_index(yp, mip_level)];

    p -= divergence[fluid.flatten_index(cell, mip_level)];

    pressure_correction_rw[fluid.flatten_index(cell, mip_level)] = p / (wxm + wxp + wym + wyp);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void fine_to_coarse(uint3 thread: SV_DispatchThreadID)
{
    uint coarse_res = fluid.resolution >> mip_level;
    if (any(thread.xy >= coarse_res) || thread.z >= 6)
        return;

    uint3 cell = thread;

    float3 dir = cube_uvw_to_dir(float3((float2(cell.xy) + 0.5) / coarse_res, cell.z));
    float3 uvw = dir_to_cube_uvw(dir);
    pressure_correction_rw[fluid.flatten_index(cell, mip_level)] = sample_cube(pressure_correction, fluid.resolution, uvw, mip_level - 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void coarse_to_fine(uint3 thread: SV_DispatchThreadID)
{
    uint fine_res = fluid.resolution >> mip_level;
    if (any(thread.xy >= fine_res) || thread.z >= 6)
        return;

    uint3 cell = thread;
    
    float3 dir = cube_uvw_to_dir(float3((float2(cell.xy) + 0.5) / fine_res, cell.z));
    float3 uvw = dir_to_cube_uvw(dir);
    pressure_correction_rw[fluid.flatten_index(cell, mip_level)] = sample_cube(pressure_correction, fluid.resolution, uvw, mip_level + 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void apply(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= fluid.resolution) || thread.z >= 6)
        return;

    uint3 cell = thread;

    uint3 xm = fluid.get_neighbor(cell, int2(-1, 0));
    uint3 xp = fluid.get_neighbor(cell, int2( 1, 0));
    uint3 ym = fluid.get_neighbor(cell, int2( 0,-1));
    uint3 yp = fluid.get_neighbor(cell, int2( 0, 1));

    float px0 = pressure_correction[fluid.flatten_index(xm)];
    float px1 = pressure_correction[fluid.flatten_index(xp)];
    float py0 = pressure_correction[fluid.flatten_index(ym)];
    float py1 = pressure_correction[fluid.flatten_index(yp)];

    // same non-uniform central difference as compute_divergence, so the
    // gradient removed here matches the spacing the divergence/Laplacian
    // were computed with
    float dist_xm = fluid.neighbor_distance(cell, xm);
    float dist_xp = fluid.neighbor_distance(cell, xp);
    float dist_ym = fluid.neighbor_distance(cell, ym);
    float dist_yp = fluid.neighbor_distance(cell, yp);

    float2 grad = float2((px1 - px0) / (dist_xm + dist_xp), (py1 - py0) / (dist_ym + dist_yp));

    fluid._velocity_rw[fluid.flatten_index(cell)] -= grad;
}
