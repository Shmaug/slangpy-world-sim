import Math;
import FluidCube;

uniform FluidCube fluid;
// mip level of pressure_correction (and divergence) that `step` relaxes this dispatch;
// `compute_divergence` and `apply` always operate at mip 0
uniform uint mip_level;

Texture2D<float>   divergence;
RWTexture2D<float> divergence_rw;
Texture2D<float>   pressure_correction;
RWTexture2D<float> pressure_correction_rw;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_divergence(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= fluid.resolution) || thread.z >= 6)
        return;

    uint3 cell = thread;
    
    uint3 xm = cube_neighbor_texel(cell, int2(-1, 0), fluid.resolution);
    uint3 xp = cube_neighbor_texel(cell, int2( 1, 0), fluid.resolution);
    uint3 ym = cube_neighbor_texel(cell, int2( 0,-1), fluid.resolution);
    uint3 yp = cube_neighbor_texel(cell, int2( 0, 1), fluid.resolution);

    let frame = CubeMapTangentFrame(cube_uvw_to_dir(float3(float2(cell.xy + 0.5) / fluid.resolution, cell.z)));
    float vel_xm = frame.to_local(fluid.load_world_space_velocity_rw(xm)).x;
    float vel_xp = frame.to_local(fluid.load_world_space_velocity_rw(xp)).x;
    float vel_ym = frame.to_local(fluid.load_world_space_velocity_rw(ym)).y;
    float vel_yp = frame.to_local(fluid.load_world_space_velocity_rw(yp)).y;

    divergence_rw[cube_to_2d(cell, fluid.resolution)] = 0.5 * ((vel_xp - vel_xm) + (vel_yp - vel_ym));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void step(uint3 thread: SV_DispatchThreadID)
{
    uint res = fluid.resolution >> mip_level;
    if (any(thread.xy >= res) || thread.z >= 6)
        return;

    uint3 cell = thread;

    float p = 0;
    p += pressure_correction.Load(uint3(cube_to_2d(cube_neighbor_texel(cell, int2(-1, 0), res), res), mip_level));
    p += pressure_correction.Load(uint3(cube_to_2d(cube_neighbor_texel(cell, int2( 1, 0), res), res), mip_level));
    p += pressure_correction.Load(uint3(cube_to_2d(cube_neighbor_texel(cell, int2( 0,-1), res), res), mip_level));
    p += pressure_correction.Load(uint3(cube_to_2d(cube_neighbor_texel(cell, int2( 0, 1), res), res), mip_level));

    p -= divergence.Load(uint3(cube_to_2d(cell, res), mip_level));

    pressure_correction_rw[cube_to_2d(cell, res)] = p * 0.25;
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
    pressure_correction_rw[cube_to_2d(cell, coarse_res)] = sample_cube(pressure_correction, uvw, mip_level - 1);
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
    pressure_correction_rw[cube_to_2d(cell, fine_res)] = sample_cube(pressure_correction, uvw, mip_level + 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void apply(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= fluid.resolution) || thread.z >= 6)
        return;

    uint3 cell = thread;

    float px0 = pressure_correction.Load(uint3(cube_to_2d(cube_neighbor_texel(cell, int2(-1, 0), fluid.resolution), fluid.resolution), 0));
    float px1 = pressure_correction.Load(uint3(cube_to_2d(cube_neighbor_texel(cell, int2( 1, 0), fluid.resolution), fluid.resolution), 0));
    float py0 = pressure_correction.Load(uint3(cube_to_2d(cube_neighbor_texel(cell, int2( 0,-1), fluid.resolution), fluid.resolution), 0));
    float py1 = pressure_correction.Load(uint3(cube_to_2d(cube_neighbor_texel(cell, int2( 0, 1), fluid.resolution), fluid.resolution), 0));

    float2 grad = 0.5 * float2(px1 - px0, py1 - py0);

    fluid.store_velocity(cell, fluid.load_velocity_rw(cell) - grad);
}
