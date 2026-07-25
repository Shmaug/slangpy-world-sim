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
    
    let frame = CubeMapTangentFrame(cube_uvw_to_dir(float3(float2(cell.xy + 0.5) / fluid.resolution, cell.z)));
    float vel_xm = frame.to_local(fluid.load_world_space_velocity_rw(fluid.get_neighbor(cell, int2(-1, 0)))).x;
    float vel_xp = frame.to_local(fluid.load_world_space_velocity_rw(fluid.get_neighbor(cell, int2( 1, 0)))).x;
    float vel_ym = frame.to_local(fluid.load_world_space_velocity_rw(fluid.get_neighbor(cell, int2( 0,-1)))).y;
    float vel_yp = frame.to_local(fluid.load_world_space_velocity_rw(fluid.get_neighbor(cell, int2( 0, 1)))).y;

    divergence_rw[flatten_cube_texel(cell, fluid.resolution)] = 0.5 * ((vel_xp - vel_xm) + (vel_yp - vel_ym));
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
    p += pressure_correction[fluid.flatten_index(fluid.get_neighbor(cell, int2(-1, 0), mip_level), mip_level)];
    p += pressure_correction[fluid.flatten_index(fluid.get_neighbor(cell, int2( 1, 0), mip_level), mip_level)];
    p += pressure_correction[fluid.flatten_index(fluid.get_neighbor(cell, int2( 0,-1), mip_level), mip_level)];
    p += pressure_correction[fluid.flatten_index(fluid.get_neighbor(cell, int2( 0, 1), mip_level), mip_level)];

    p -= divergence[fluid.flatten_index(cell, mip_level)];

    pressure_correction_rw[fluid.flatten_index(cell, mip_level)] = p * 0.25;
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

    float px0 = pressure_correction[fluid.flatten_index(fluid.get_neighbor(cell, int2(-1, 0)))];
    float px1 = pressure_correction[fluid.flatten_index(fluid.get_neighbor(cell, int2( 1, 0)))];
    float py0 = pressure_correction[fluid.flatten_index(fluid.get_neighbor(cell, int2( 0,-1)))];
    float py1 = pressure_correction[fluid.flatten_index(fluid.get_neighbor(cell, int2( 0, 1)))];

    float2 grad = 0.5 * float2(px1 - px0, py1 - py0);

    fluid._velocity_rw[fluid.flatten_index(cell)] -= grad;
}
