Texture2D<float> velocity_x;
Texture2D<float> velocity_y;

RWTexture2D<float> velocity_x_out;
RWTexture2D<float> velocity_y_out;
RWTexture2D<float> cell_p;

RWTexture2D<float> cell_r;
RWTexture2D<float> cell_d;
RWTexture2D<float> cell_q;
RWTexture2D<float> r_dot_r;
RWTexture2D<float> d_dot_q;

Texture2D<float> r_dot_r_view;
Texture2D<float> d_dot_q_view;
Texture2D<float> prev_sigma;

uniform uint2 dim;

[shader("compute")]
[numthreads(8, 8, 1)]
void calculate_residuals(uint3 index: SV_DispatchThreadID)
{
    int2 cell = index.xy;
    if (any(cell >= dim))
        return;

    float vx0 = cell.x == 0       ? 0 : velocity_x[cell];
    float vx1 = cell.x == dim.x-1 ? 0 : velocity_x[cell + int2(1, 0)];
    float vy0 = cell.y == 0       ? 0 : velocity_y[cell];
    float vy1 = cell.y == dim.y-1 ? 0 : velocity_y[cell + int2(0, 1)];

    float r = -(vx1 - vx0 + vy1 - vy0);

    cell_r[cell] = r;
    r_dot_r[cell] = r * r;

    cell_d[cell] = r;
    cell_p[cell] = 0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void mul_A(uint3 index: SV_DispatchThreadID)
{
    int2 cell = index.xy;
    if (any(cell >= dim))
        return;

    float q = cell_d[cell];
    if (cell.x > 0) q -= cell_d[cell - int2(1, 0)];
    if (cell.y > 0) q -= cell_d[cell - int2(0, 1)];
    if (cell.x < dim.x-1) q -= cell_d[cell + int2(1, 0)];
    if (cell.y < dim.y-1) q -= cell_d[cell + int2(0, 1)];

    cell_q[cell] = q;

    d_dot_q[cell] = cell_d[cell] * q;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void update_residuals(uint3 index: SV_DispatchThreadID)
{
    int2 cell = index.xy;
    if (any(cell >= dim))
        return;

    float alpha = r_dot_r_view[0] / d_dot_q_view[0];
    cell_p[cell] += alpha * cell_d[cell];

    float r = cell_r[cell] - alpha * cell_q[cell];
    cell_r[cell] = r;
    r_dot_r[cell] = r * r;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void update_cell_d(uint3 index: SV_DispatchThreadID)
{
    int2 cell = index.xy;
    if (any(cell >= dim))
        return;

    float beta = r_dot_r_view[0] / prev_sigma[0];
    cell_d[cell] = cell_r[cell] + beta * cell_d[cell];    
}

[shader("compute")]
[numthreads(8,8,1)]
void apply(uint3 index : SV_DispatchThreadID)
{
    int2 cell = index.xy;
    if (any(cell >= dim))
        return;
    
    float vx = 0;
    float vy = 0;

    float d = cell_p[cell];
    if (cell.x > 0) vx = velocity_x[cell] - (d - cell_p[cell + int2(-1,  0)]);
    if (cell.y > 0) vy = velocity_y[cell] - (d - cell_p[cell + int2( 0, -1)]);

    velocity_x_out[cell] = vx;
    velocity_y_out[cell] = vy;

    // zero boundary velocities
    if (cell.x == dim.x-1) velocity_x_out[cell + int2(1,0)] = 0;
    if (cell.y == dim.y-1) velocity_y_out[cell + int2(0,1)] = 0;
}