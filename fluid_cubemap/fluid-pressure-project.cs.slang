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

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_divergence(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= velocity.resolution) || thread.z >= 6)
        return;

    uint3 cell = thread;

    uint3 xm = velocity.get_neighbor(cell, int2(-1, 0));
    uint3 xp = velocity.get_neighbor(cell, int2( 1, 0));
    uint3 ym = velocity.get_neighbor(cell, int2( 0,-1));
    uint3 yp = velocity.get_neighbor(cell, int2( 0, 1));

    float3 dir = cube_uvw_to_dir(float3(float2(cell.xy + 0.5) / velocity.resolution, cell.z));
    let frame = CubeMapTangentFrame(dir);
    float vel_xm = frame.to_local(velocity.load_world_space(xm, dir)).x;
    float vel_xp = frame.to_local(velocity.load_world_space(xp, dir)).x;
    float vel_ym = frame.to_local(velocity.load_world_space(ym, dir)).y;
    float vel_yp = frame.to_local(velocity.load_world_space(yp, dir)).y;

    // non-uniform central difference: neighbor spacing varies with where
    // `cell` sits on its face, so the two half-differences are weighted by
    // the actual geodesic distance to each neighbor instead of assumed to
    // both be one texel wide
    float dist_xm = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(xm.xy + 0.5) / velocity.resolution, xm.z)));
    float dist_xp = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(xp.xy + 0.5) / velocity.resolution, xp.z)));
    float dist_ym = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(ym.xy + 0.5) / velocity.resolution, ym.z)));
    float dist_yp = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(yp.xy + 0.5) / velocity.resolution, yp.z)));

    divergence.store(cell, (vel_xp - vel_xm) / (dist_xm + dist_xp) + (vel_yp - vel_ym) / (dist_ym + dist_yp));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void step(uint3 thread: SV_DispatchThreadID)
{
    uint res = velocity.resolution >> dst_mip_level;
    if (any(thread.xy >= res) || thread.z >= 6)
        return;

    uint3 cell = thread;

    uint3 xm = divergence.get_neighbor(cell, int2(-1, 0), dst_mip_level);
    uint3 xp = divergence.get_neighbor(cell, int2( 1, 0), dst_mip_level);
    uint3 ym = divergence.get_neighbor(cell, int2( 0,-1), dst_mip_level);
    uint3 yp = divergence.get_neighbor(cell, int2( 0, 1), dst_mip_level);

    // non-uniform Jacobi stencil for the Laplacian: on a uniform grid every
    // neighbor's weight collapses to 1 (matching the old p*0.25 average),
    // but here each axis's pair of neighbor distances can differ, so the
    // weights are derived from the standard unequal-spacing second-
    // derivative finite-difference formula instead of a plain average
    float3 dir = cube_uvw_to_dir(float3(float2(cell.xy + 0.5) / velocity.resolution, cell.z));
    uint mip_res = velocity.resolution >> dst_mip_level;
    float hxm = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(xm.xy + 0.5) / mip_res, xm.z)));
    float hxp = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(xp.xy + 0.5) / mip_res, xp.z)));
    float hym = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(ym.xy + 0.5) / mip_res, ym.z)));
    float hyp = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(yp.xy + 0.5) / mip_res, yp.z)));

    float wxm = 2.0 / (hxm * (hxm + hxp));
    float wxp = 2.0 / (hxp * (hxm + hxp));
    float wym = 2.0 / (hym * (hym + hyp));
    float wyp = 2.0 / (hyp * (hym + hyp));

    float p = 0;
    p += wxm * pressure_correction[0].load(xm, dst_mip_level);
    p += wxp * pressure_correction[0].load(xp, dst_mip_level);
    p += wym * pressure_correction[0].load(ym, dst_mip_level);
    p += wyp * pressure_correction[0].load(yp, dst_mip_level);

    p -= divergence.load(cell, dst_mip_level);

    pressure_correction[1].store(cell, p / (wxm + wxp + wym + wyp), dst_mip_level);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resample(uint3 thread: SV_DispatchThreadID)
{
    uint dst_res = velocity.resolution >> dst_mip_level;
    if (any(thread.xy >= dst_res) || thread.z >= 6)
        return;

    uint3 cell = thread;
    float3 dir = cube_uvw_to_dir(float3((float2(cell.xy) + 0.5) / dst_res, cell.z));

    pressure_correction[1].store(cell, pressure_correction[0].sample(dir, src_mip_level), dst_mip_level);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void apply(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= velocity.resolution) || thread.z >= 6)
        return;

    uint3 cell = thread;

    uint3 xm = pressure_correction[0].get_neighbor(cell, int2(-1, 0));
    uint3 xp = pressure_correction[0].get_neighbor(cell, int2( 1, 0));
    uint3 ym = pressure_correction[0].get_neighbor(cell, int2( 0,-1));
    uint3 yp = pressure_correction[0].get_neighbor(cell, int2( 0, 1));

    float px0 = pressure_correction[0].load(xm);
    float px1 = pressure_correction[0].load(xp);
    float py0 = pressure_correction[0].load(ym);
    float py1 = pressure_correction[0].load(yp);

    // same non-uniform central difference as compute_divergence, so the
    // gradient removed here matches the spacing the divergence/Laplacian
    // were computed with
    float3 dir = cube_uvw_to_dir(float3(float2(cell.xy + 0.5) / velocity.resolution, cell.z));
    float dist_xm = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(xm.xy + 0.5) / velocity.resolution, xm.z)));
    float dist_xp = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(xp.xy + 0.5) / velocity.resolution, xp.z)));
    float dist_ym = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(ym.xy + 0.5) / velocity.resolution, ym.z)));
    float dist_yp = sphere_geodesic_distance(dir, cube_uvw_to_dir(float3(float2(yp.xy + 0.5) / velocity.resolution, yp.z)));

    float2 grad = float2((px1 - px0) / (dist_xm + dist_xp), (py1 - py0) / (dist_ym + dist_yp));

    velocity.store(cell, velocity.load(cell) - grad);
}
