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
uniform int      render_channel;

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
                    if (smoke.channels == 1 || render_channel >= 0 && render_channel < smoke.channels)
                    {
                        uint ch = render_channel >= 0 && render_channel < smoke.channels ? render_channel : 0;
                        float s = linear_sampling == 0 ? smoke.sample_layer_nearest<1>(dir, ch, 0, mip_level) : smoke.sample_layer<1>(dir, ch, 0, mip_level);
                        c.rgb = saturate(float3(1 - exp(s), 1 - exp(-s), 0.1));
                    }
                    else if (smoke.channels == 2)
                    {
                        float2 s = linear_sampling == 0 ? smoke.sample_layer_nearest<2>(dir, 0, 0, mip_level) : smoke.sample_layer<2>(dir, 0, 0, mip_level);
                        c.rgb = saturate(float3(s, 0.1));
                    }
                    else if (smoke.channels >= 3)
                    {
                        float3 s = linear_sampling == 0 ? smoke.sample_layer_nearest<3>(dir, 0, 0, mip_level) : smoke.sample_layer<3>(dir, 0, 0, mip_level);
                        c.rgb = saturate(0.1 + s);
                    }
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

                float optical_depth = 0;
                float t = atmosphere_t.x;
                for (uint i = 0; i < N; i++)
                {
                    float3 pos = camera_pos + ray_dir * t;
                    if (t >= atmosphere_t.y)
                        break;

                    float3 color;
                    float density;
                    if ((RenderMode)render_mode == RenderMode::Smoke)
                    {
                        float3 s;
                        if (smoke.channels == 1 || render_channel >= 0 && render_channel < smoke.channels)
                        {
                            uint ch = render_channel >= 0 && render_channel < smoke.channels ? render_channel : 0;
                            s.x = linear_sampling == 0 ? smoke.sample_nearest<1>(pos, ch, mip_level) : smoke.sample<1>(pos, ch, mip_level);
                            s = 0.1 + float3(1 - exp(s.x), 1 - exp(-s.x), 0.1);
                        }
                        else if (smoke.channels == 2)
                        {
                            s.xy = linear_sampling == 0 ? smoke.sample_nearest<2>(pos, 0, mip_level) : smoke.sample<2>(pos, 0, mip_level);
                            s.z = 0.1;
                        }
                        else if (smoke.channels >= 3)
                        {
                            s = 0.1 + (linear_sampling == 0 ? smoke.sample_nearest<3>(pos, 0, mip_level) : smoke.sample<3>(pos, 0, mip_level));
                        }
                        density = 0.1 + dot(s,1);
                    }
                    else
                    {
                        float3 v = linear_sampling == 0 ? velocity.sample_nearest(pos) : velocity.sample(pos);
                        if ((RenderMode)render_mode == RenderMode::TangentVelocity)
                            v = velocity.to_local(normalize(pos), v);
                        color = 5 * v + 0.5;
                        density = 0.1 + length(v);
                    }

                    density *= velocity.sample_density(length(pos));
                    density *= 5 / velocity.thickness;

                    float step = (2 * (1 + velocity.thickness)) / float(N);
                    step *= 0.5 + rng.nextFloat() * exp(-density);
                    step = min(step, atmosphere_t.y - t);

                    c.rgb += color * density * step * exp(-optical_depth);
                    optical_depth += step * density;

                    t += step;
                }
            }
        }
    }

    image[pixel] = c;
}