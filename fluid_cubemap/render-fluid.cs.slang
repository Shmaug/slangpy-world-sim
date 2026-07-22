import Math;
import FluidCube;

uniform FluidCube fluid;

uniform float4x4 view_projection;
uniform float4x4 inv_view_projection;
uniform float3   camera_pos;
uniform uint     render_mode;
uniform uint2    resolution;
uniform float2   mouse_pos;

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
                float smoke = fluid.sample_smoke(pos);
                c.rgb = saturate(float3(1 - exp(smoke), 1 - exp(-smoke), 0.1));
                break;
            }
            case RenderMode::Velocity:
            {
                float3 vel = fluid.sample_velocity(pos);
                c.rgb = 5 * vel + 0.5;
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
                uint3 texel = uint3(uint2(uvw.xy * fluid.resolution), uint(uvw.z));

                c.rgb = float3(uint3(pcg(texel.x + pcg(texel.y)), pcg(texel.y + pcg(texel.x)), 0) % 255) / 255;
                c.rgb = lerp(c.rgb, 0.5, 0.75);

                float3 mouse_ray_dir = get_ray_dir(mouse_pos);
                float2 mouse_t;
                if (intersect_sphere(camera_pos, mouse_ray_dir, 1, mouse_t))
                {
                    float3 mouse_pos = camera_pos + mouse_ray_dir * mouse_t.x;
                    float3 mouse_uvw = dir_to_cube_uvw(mouse_pos);
                    uint3 mouse_texel = uint3(uint2(mouse_uvw.xy * fluid.resolution), uint(mouse_uvw.z));
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
                        uint3 neighbor_texel = fluid.neighbor_cell(texel, neighbors[i]);
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
    image[pixel] = c;
}