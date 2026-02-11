RWTexture2D<float> density;
RWTexture2D<float> velocity_x;
RWTexture2D<float> velocity_y;

uniform uint2 dim;
uniform float2 drag_start;
uniform float2 drag_stop;
uniform float drag_radius;
uniform float drag_velocity;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 index : SV_DispatchThreadID)
{
    uint2 pos = index.xy;
    if (any(pos >= dim))
        return;

    float d = 0;
    float2 v = 0;

    //*
    float2 to_c = pos - dim * 0.5;

    float r = max(dim.x, dim.y) * 0.25;
    if (length(to_c) < r)
    {
        if (abs(to_c.x) < r * .01 || abs(to_c.y) < r * .01)
            d = 1;
    }

    //v = 4 * float2(to_c.y, -to_c.x) / max(dim.x, dim.y);

    /*/
    const float r = max(dim.x, dim.y)/10.0;
    const float2 s0 = dim * 0.5 - float2(r * 1.5, r * 0.5);
    const float2 s1 = dim - s0;

    float2 d0 = (pos + 0.5) - (s0 + 0.5);
    float2 d1 = (pos + 0.5) - (s1 + 0.5);

    if (dot(d0, d0) < r*r)
    {
        v = float2(1, 0);// + 0.25 * float2(-d0.y, -d0.x) / r;
        d = 1;
    }
    if (dot(d1, d1) < r*r)
    {
        v = float2(-1, 0);// - 0.25 * float2(-d1.y, -d1.x) / r;
        d = 1;
    }
    //*/

    density[pos] = d;
    velocity_x[pos] = pos.x == 0 ? 0 : v.x;
    velocity_y[pos] = pos.y == 0 ? 0 : v.y;
    if (pos.x == dim.x-1) velocity_x[pos + int2(1,0)] = 0;
    if (pos.y == dim.y-1) velocity_y[pos + int2(0,1)] = 0;
}

float distance2_to_drag(float2 pos)
{
    float2 drag_dir = drag_stop - drag_start;
    float drag_length = length(drag_dir);
    drag_dir /= drag_length;
    
    float drag_t = saturate(dot(drag_dir, pos - drag_start) / drag_length);
    float2 pos_on_line = drag_start + drag_dir * drag_t;
    float2 d = pos - pos_on_line;
    return dot(d, d);
}

[shader("compute")]
[numthreads(8,8,1)]
void drag(uint3 index : SV_DispatchThreadID)
{
    uint2 pos = index.xy;

    // add density to cells within radius
    if (all(pos < dim))
    {
        float r2 = distance2_to_drag(pos + 0.5) / (drag_radius*drag_radius);
        if (r2 < 1.0)
        {
            // density[pos] += exp(-4 * r2);
        }
    }

    float2 drag_dir = drag_stop - drag_start;
    float drag_length = length(drag_dir);
    drag_dir /= drag_length;

    // add velocity to edges within radius

    // top edge of cell
    if (pos.y > 0 && pos.y < dim.y)
    {
        float r2 = distance2_to_drag(pos + float2(0.5, 0)) / (drag_radius * drag_radius);
        if (r2 < 1.0)
        {
            velocity_y[pos] += drag_dir.y * drag_velocity * drag_length * exp(-8*r2);
        }
    }

    // left edge of cell
    if (pos.x > 0 && pos.x < dim.x)
    {
        float r2 = distance2_to_drag(pos + float2(0.0, 0.5)) / (drag_radius * drag_radius);
        if (r2 < 1.0)
        {
            velocity_x[pos] += drag_dir.x * drag_velocity * drag_length * exp(-8*r2);
        }
    }
}