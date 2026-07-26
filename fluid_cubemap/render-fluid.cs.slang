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

RWTexture2D<float4> image;

enum RenderMode
{
    Smoke,
    Velocity,
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
        float3 pos = normalize(camera_pos + ray_dir * ray_t.x);

        let frame = CubeMapTangentFrame(pos);

        switch ((RenderMode)render_mode)
        {
            case RenderMode::Smoke:
            {
                if (smoke.vertical_resolution == 1)
                {
                    float s = smoke.sample_layer(pos, mip_level, 0);
                    c.rgb = saturate(float3(1 - exp(s), 1 - exp(-s), 0.1));
                }
                break;
            }
            case RenderMode::Velocity:
            {
                if (smoke.vertical_resolution == 1)
                {
                    float3 v = velocity.sample_layer(pos, 0);
                    c.rgb = 5 * v + 0.5;
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
                float3 uvw = dir_to_cube_uvw(pos);
                uint3 texel = uint3(uint2(uvw.xy * smoke.resolution), uint(uvw.z));

                c.rgb = float3(uint3(pcg(texel.x + pcg(texel.y)), pcg(texel.y + pcg(texel.x)), 0) % 255) / 255;
                c.rgb = lerp(c.rgb, 0.5, 0.75);

                float3 mouse_ray_dir = get_ray_dir(mouse_pos);
                float2 mouse_t;
                if (intersect_sphere(camera_pos, mouse_ray_dir, 1, mouse_t))
                {
                    float3 mouse_pos = camera_pos + mouse_ray_dir * mouse_t.x;
                    float3 mouse_uvw = dir_to_cube_uvw(mouse_pos);
                    uint3 mouse_texel = uint3(uint2(mouse_uvw.xy * smoke.resolution), uint(mouse_uvw.z));
                    if (all(mouse_texel == texel))
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
                        uint3 neighbor_texel = cube_neighbor_texel(texel, neighbors[i], smoke.resolution);
                        if (all(mouse_texel == neighbor_texel))
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
            if ((RenderMode)render_mode == RenderMode::Smoke || (RenderMode)render_mode == RenderMode::Velocity)
            {
                atmosphere_t.x = max(0, atmosphere_t.x);
                if (ray_t.x > 0)
                    atmosphere_t.y = min(ray_t.x, atmosphere_t.y);

                RNG rng = RNG(asuint(atmosphere_t.y) + pcg(asuint(atmosphere_t.x)));

                uint N = 64;
                float optical_depth = 0;
                float t = atmosphere_t.x;
                for (uint i = 0; i < N; i++)
                {
                    float3 pos = camera_pos + ray_dir * t;
                    if (t > atmosphere_t.y)
                        break;

                    float3 color;
                    float density;
                    if ((RenderMode)render_mode == RenderMode::Smoke)
                    {
                        float s = smoke.sample(pos, mip_level);
                        color = saturate(float3(1 - exp(s), 1 - exp(-s), 0.1));
                        density = s;
                    }
                    else
                    {
                        float3 v = velocity.sample(pos);
                        color = 5 * v + 0.5;
                        density = length(v);
                    }

                    density = 1 + 5*density;
                    density *= velocity.sample_density(length(pos)) / velocity.thickness;

                    float step = (rng.nextFloat() + 5 * exp(-density)) * (atmosphere_t.y - atmosphere_t.x) / float(N);

                    c.rgb += 3 * color * density * step * exp(-optical_depth);
                    optical_depth += step * density;

                    t += step;
                }
            }
        }
    }

    image[pixel] = c;
}