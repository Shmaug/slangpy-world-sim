import Math;
import FluidGrid;

uniform FluidGrid grid;
RWTexture2D<float> curl; // computed at top-left corner of each grid cell
uniform float confinement_scale;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_curl(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution))
        return;

    //         |
    //        vx0
    //         |
    // --vy0-- * --vy1--
    //         |
    //        vx1   cell
    //         |

    float vx0 = grid.velocity_x_rw[wrap_index(int2(cell) + int2(0, -1), grid.resolution + uint2(1, 0))];
    float vx1 = grid.velocity_x_rw[cell];
    float vy0 = grid.velocity_y_rw[wrap_index(int2(cell) + int2(-1, 0), grid.resolution + uint2(0, 1))];
    float vy1 = grid.velocity_y_rw[cell];
    
    curl[cell] = (vy1 - vy0) - (vx1 - vx0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void add_confinement_force(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution+1))
        return;

    float L = curl[wrap_index(cell + int2(-1,  0), grid.resolution)];
    float R = curl[wrap_index(cell + int2( 1,  0), grid.resolution)];
    float T = curl[wrap_index(cell + int2( 0,  1), grid.resolution)];
    float B = curl[wrap_index(cell + int2( 0, -1), grid.resolution)];
    float C = curl[cell];

    float2 N = float2(abs(T) - abs(B), abs(R) - abs(L));
    N = safe_normalize(N);

    float2 force = confinement_scale * C;

    grid.velocity_x_rw[cell] += force.x * C;
    grid.velocity_y_rw[cell] += force.y * C;
}