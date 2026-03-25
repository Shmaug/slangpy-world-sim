import Math;
import FluidGrid;

uniform FluidGrid grid;

[shader("compute")]
[numthreads(8, 8, 1)]
void init_vortex(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution+1))
        return;

    const float2 circles[2] = {
        float2(.5, .5) * grid.resolution,
        float2(.75, .5) * grid.resolution,
    };
    const float radius = 0.3 * min(grid.resolution.x, grid.resolution.y);

    float pressure = 0;
    float2 vel = 0;

    for (uint i = 0; i < 1; i++)
    {
        float2 s = (cell + 0.5) - circles[i];
        
        if (length(s) < radius)
        {
            pressure += 0.5;
        }

        vel += float2(-s.y, s.x) * 0.1;
    }

    if (all(cell < grid.resolution))
        grid.pressure_rw[cell] = pressure;
    if (cell.y < grid.resolution.y)
        grid.velocity_x_rw[cell] = vel.x;
    if (cell.x < grid.resolution.x)
        grid.velocity_y_rw[cell] = vel.y;
}

// Leapfrogging vortex rings: two co-moving vortex pairs that periodically
// pass through each other. Each pair consists of two opposite-sign point
// vortices offset vertically, producing net rightward motion. The two pairs
// are placed one behind the other so the rear pair accelerates through the
// front, creating the classic leapfrog dynamics.
[shader("compute")]
[numthreads(8, 8, 1)]
void init_leapfrog_vortex(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution + 1))
        return;

    float2 res = float2(grid.resolution);
    float2 pos = float2(cell) + 0.5;

    // Four vortices: two pairs, each pair has +/- circulation
    // Pair A (leading) and Pair B (trailing), separated horizontally
    float cy = res.y * 0.5;           // vertical center
    float sep = res.y * 0.06;         // half-separation between +/- vortices in a pair
    float pair_gap = res.x * 0.08;    // horizontal distance between the two pairs
    float cx = res.x * 0.5;           // horizontal center

    // vortex positions: (x, y, sign of circulation)
    //   Pair A (front): positive on top, negative on bottom -> net rightward
    //   Pair B (rear):  same arrangement
    float3 vortices[4] = {
        float3(cx + pair_gap, cy - sep,  1.0),   // A top  (+)
        float3(cx + pair_gap, cy + sep, -1.0),   // A bot  (-)
        float3(cx - pair_gap, cy - sep,  1.0),   // B top  (+)
        float3(cx - pair_gap, cy + sep, -1.0),   // B bot  (-)
    };

    float strength = 0.15 * min(res.x, res.y);
    float core_radius = res.y * 0.03;   // soft-core radius to avoid singularity

    float pressure = 0;
    float2 vel = 0;

    for (uint i = 0; i < 4; i++)
    {
        float2 center = vortices[i].xy;
        float sign = vortices[i].z;

        float2 r = pos - center;
        // periodic wrap for shortest distance
        r -= round(r / res) * res;

        float dist2 = dot(r, r);
        float dist = sqrt(dist2);

        // Lamb-Oseen style smooth vortex: v_theta = Gamma/(2*pi*r) * (1 - exp(-r^2/rc^2))
        float rc2 = core_radius * core_radius;
        float factor = sign * strength / (M_2PI * max(dist, 1e-6)) * (1.0 - exp(-dist2 / rc2));

        vel += float2(-r.y, r.x) * factor;

        // Pressure ~ -0.5 * |vel|^2 from Bernoulli; approximate with Gaussian blob
        pressure += sign * 0.5 * exp(-dist2 / (2.0 * rc2));
    }

    if (all(cell < grid.resolution))
        grid.pressure_rw[cell] = pressure;
    if (cell.y < grid.resolution.y)
        grid.velocity_x_rw[cell] = vel.x;
    if (cell.x < grid.resolution.x)
        grid.velocity_y_rw[cell] = vel.y;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void emit_smoke(uint3 thread: SV_DispatchThreadID)
{
    uint2 cell = thread.xy;
    if (any(cell >= grid.resolution + 1))
        return;

    static const float4 box = float4(
        grid.resolution.x * 0.25, grid.resolution.y * 0.8,
        grid.resolution.x * 0.75, grid.resolution.y * 0.9);

    float pressure = grid.pressure[cell];
    float2 vel = float2(grid.velocity_x[cell], grid.velocity_y[cell]);

    if (all(cell >= box.xy) && all(cell < box.zw))
    {
        pressure = 1;
        vel = float2(0, -0.1) * grid.resolution;
    }

    if (all(cell < grid.resolution))
        grid.pressure_rw[cell] = pressure;
    if (cell.y < grid.resolution.y)
        grid.velocity_x_rw[cell] = vel.x;
    if (cell.x < grid.resolution.x)
        grid.velocity_y_rw[cell] = vel.y;
}
