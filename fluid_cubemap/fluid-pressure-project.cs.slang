import Math;
import FluidCube;

uniform FluidCube fluid;
RWTexture2D<float> divergence;
Texture2D<float>   pressure_correction;
RWTexture2D<float> pressure_correction_rw;

uint2 cube_to_2d(uint3 texel)
{
    return uint2(texel.xy + uint2(texel.z * fluid.resolution, 0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_divergence(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= fluid.resolution) || thread.z >= 6)
        return;

    uint3 cell = thread;
    
    uint3 xm = fluid.neighbor_cell(cell, int2(-1, 0));
    uint3 xp = fluid.neighbor_cell(cell, int2( 1, 0));
    uint3 ym = fluid.neighbor_cell(cell, int2( 0,-1));
    uint3 yp = fluid.neighbor_cell(cell, int2( 0, 1));

    let frame = CubeMapTangentFrame(cube_uvw_to_dir(float3(float2(cell.xy + 0.5) / fluid.resolution, cell.z)));
    float vel_xm = frame.to_local(fluid.load_world_space_velocity_rw(xm)).x;
    float vel_xp = frame.to_local(fluid.load_world_space_velocity_rw(xp)).x;
    float vel_ym = frame.to_local(fluid.load_world_space_velocity_rw(ym)).y;
    float vel_yp = frame.to_local(fluid.load_world_space_velocity_rw(yp)).y;

    divergence[cube_to_2d(cell)] = 0.5 * ((vel_xp - vel_xm) + (vel_yp - vel_ym));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void step(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= fluid.resolution) || thread.z >= 6)
        return;

    uint3 cell = thread;

    float p = 0;
    p += pressure_correction[cube_to_2d(fluid.neighbor_cell(cell, int2(-1, 0)))];
    p += pressure_correction[cube_to_2d(fluid.neighbor_cell(cell, int2( 1, 0)))];
    p += pressure_correction[cube_to_2d(fluid.neighbor_cell(cell, int2( 0,-1)))];
    p += pressure_correction[cube_to_2d(fluid.neighbor_cell(cell, int2( 0, 1)))];

    p -= divergence[cube_to_2d(cell)];

    pressure_correction_rw[cube_to_2d(cell)] = p * 0.25;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void apply(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= fluid.resolution) || thread.z >= 6)
        return;

    uint3 cell = thread;

    float px0 = pressure_correction[cube_to_2d(fluid.neighbor_cell(cell, int2(-1, 0)))];
    float px1 = pressure_correction[cube_to_2d(fluid.neighbor_cell(cell, int2( 1, 0)))];
    float py0 = pressure_correction[cube_to_2d(fluid.neighbor_cell(cell, int2( 0,-1)))];
    float py1 = pressure_correction[cube_to_2d(fluid.neighbor_cell(cell, int2( 0, 1)))];

    float2 grad = 0.5 * float2(px1 - px0, py1 - py0);

    fluid.store_velocity(cell, fluid.load_velocity_rw(cell) - grad);
}
