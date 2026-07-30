import Math;
import SphericalField;
import RNG;

uniform SphericalScalarField smoke;
uniform SphericalVelocityField velocity;

uniform float4x4 view_projection;
uniform float4x4 inv_view_projection;
uniform float3   camera_pos;
uniform uint     render_mode;
uniform uint2    resolution;
uniform float2   mouse_pos;
uniform uint     mip_level;
uniform uint     frame_index;
uniform uint     linear_sampling;

RWTexture2D<float4> image;

enum RenderMode
{
    Smoke,
    Velocity,
    TangentVelocity,
    Tangent,
    Bitangent,
    Texels
}

float3 get_ray_dir(float2 coord)
{
    float2 uv = coord / resolution;
    float2 clip_pos = 2 * uv - 1;
    clip_pos.y = -clip_pos.y;
    return normalize(mul(inv_view_projection, float4(clip_pos, 1, 1)).xyz);
}

float3 plasma_colormap(float t)
{
    float3 c0 = float3(0.05873234392399702, 0.02333670892565664, 0.5433401826748754);
    float3 c1 = float3(2.176514634195958, 0.2383834171260182, 0.7539604599784036);
    float3 c2 = float3(-2.689460476458034, -7.455851135738909, 3.110799939717086);
    float3 c3 = float3(6.130348345893603, 42.3461881477227, -28.51885465332158);
    float3 c4 = float3(-11.10743619062271, -82.66631109428045, 60.13984767418263);
    float3 c5 = float3(10.02306557647065, 71.41361770095349, -54.07218655560067);
    float3 c6 = float3(-3.658713842777788, -22.93153465461149, 18.19190778539828);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}

[shader("compute")]
[numthreads(8,8,1)]
void render(uint2 pixel: SV_DispatchThreadID)
{
    if (any(pixel >= resolution))
        return;

    float4 c = float4(0, 0, 0, 1);

    float3 ray_dir = get_ray_dir(pixel + 0.5);

    float2 ray_t;
    if (intersect_sphere(camera_pos, ray_dir, 1, ray_t))
    {
        float3 dir = normalize(camera_pos + ray_dir * ray_t.x);

        let frame = CubeMapTangentFrame(dir);

        switch ((RenderMode)render_mode)
        {
            case RenderMode::Smoke:
            {
                if (smoke.vertical_resolution == 1)
                {
                    float s = linear_sampling == 0 ? smoke.sample_layer_nearest<1>(dir, 0, 0, mip_level) : smoke.sample_layer<1>(dir, 0, 0, mip_level);
                    c.rgb = plasma_colormap(saturate(s));
                }
                break;
            }
            case RenderMode::Velocity:
            case RenderMode::TangentVelocity:
            {
                if (smoke.vertical_resolution == 1)
                {
                    float3 v = linear_sampling == 0 ? velocity.sample_layer_nearest(dir, 0) : velocity.sample_layer(dir, 0);
                    if ((RenderMode)render_mode == RenderMode::TangentVelocity)
                        v = velocity.to_local(dir, v);
                    c.rgb = v + 0.5;
                }
                break;
            }
            case RenderMode::Tangent:
            {
                c.rgb = frame.t * 0.5 + 0.5;
                break;
            }
            case RenderMode::Bitangent:
            {
                c.rgb = frame.b * 0.5 + 0.5;
                break;
            }
            case RenderMode::Texels:
            {
                float3 uvw = dir_to_cube_uvw(dir);
                CubeCoord coord = CubeCoord(uint2(uvw.xy * smoke.resolution), uint(uvw.z), 0, 0);

                c.rgb = float3(uint3(pcg(coord.texel.x + pcg(coord.texel.y)), pcg(coord.texel.y + pcg(coord.texel.x)), 0) % 255) / 255;
                c.rgb = lerp(c.rgb, 0.5, 0.75);

                float3 mouse_ray_dir = get_ray_dir(mouse_pos);
                float2 mouse_t;
                if (intersect_sphere(camera_pos, mouse_ray_dir, 1, mouse_t))
                {
                    float3 mouse_pos = camera_pos + mouse_ray_dir * mouse_t.x;
                    float3 mouse_uvw = dir_to_cube_uvw(mouse_pos);
                    CubeCoord mouse_coord = CubeCoord(uint2(mouse_uvw.xy * smoke.resolution), uint(mouse_uvw.z), 0, 0);
                    if (all(mouse_coord.texel == coord.texel) && mouse_coord.face == coord.face)
                    {
                        c.rgb = 1;
                    }

                    int2 neighbors[] = {
                        int2(-1, 0),
                        int2(1, 0),
                        int2(0, -1),
                        int2(0, 1),
                    };
                    for (uint i = 0; i < 4; i++)
                    {
                        CubeCoord neighbor_coord = cube_neighbor_texel(coord, neighbors[i], smoke.resolution);
                        if (all(mouse_coord.texel == neighbor_coord.texel) && mouse_coord.face == neighbor_coord.face)
                        {
                            c.rgb = float3(1, 0, 0);
                        }
                    }
                }

                break;
            }
        }
    }

    if (smoke.vertical_resolution > 1)
    {
        float2 atmosphere_t;
        if (intersect_sphere(camera_pos, ray_dir, 1 + smoke.thickness, atmosphere_t))
        {
            if ((RenderMode)render_mode == RenderMode::Smoke || (RenderMode)render_mode == RenderMode::Velocity || (RenderMode)render_mode == RenderMode::TangentVelocity)
            {
                atmosphere_t.x = max(0, atmosphere_t.x);
                if (ray_t.x > 0)
                    atmosphere_t.y = min(ray_t.x, atmosphere_t.y);

                RNG rng = RNG(pcg(frame_index + pcg(pixel.y + pcg(pixel.x))));

                uint N = 64;

                float texel_solid_angle = M_4PI / float(smoke.resolution * smoke.resolution * 6);
                float min_step = 0.5 * sqrt(texel_solid_angle);
                
                float optical_depth = 0;
                float t = atmosphere_t.x;
                float4 prev_sample = 0;

                t += min_step * rng.nextFloat();

                bool done = false;
                for (uint i = 0; i < N && !done; i++)
                {
                    float step = min_step * (1 + 10*rng.nextFloat() * exp(-prev_sample.w/50));
                    if (step >= atmosphere_t.y - t)
                    {
                        step = atmosphere_t.y - t;
                        t = atmosphere_t.y;
                        done = true;
                    }
                    else
                    {
                        t += step;
                    }
                    
                    float3 pos = camera_pos + ray_dir * t;

                    float4 sample;
                    if ((RenderMode)render_mode == RenderMode::Smoke)
                    {
                        float s = linear_sampling == 0 ? smoke.sample_nearest<1>(pos, 0, mip_level) : smoke.sample<1>(pos, 0, mip_level);
                        sample.rgb = plasma_colormap(saturate(s));
                        sample.w = 0.5 + 50*s;
                    }
                    else
                    {
                        float3 v = linear_sampling == 0 ? velocity.sample_nearest(pos) : velocity.sample(pos);
                        if ((RenderMode)render_mode == RenderMode::TangentVelocity)
                            v = velocity.to_local(normalize(pos), v);
                        sample.rgb = 5 * v + 0.5;
                        sample.w = 10 * length(v);
                    }

                    float d = smoke.sample_density(length(pos));
                    sample.w *= d;
                    sample.w *= step;

                    // trapezoidal integration
                    c.rgb += sample.rgb * sample.w * exp(-optical_depth);
                    optical_depth += sample.w;

                    prev_sample = sample;
                }
            }
        }
    }

    image[pixel] = c;
}