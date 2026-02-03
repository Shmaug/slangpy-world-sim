#include "Math.slangh"

static const float kEarthRadius = 6371; // kilometers
static const float kTerrainAltitude = 6.4 / kEarthRadius;
static const float2 kCloudAltitudeRange = float2(2, 10) / kEarthRadius;
static const float3 kSunDirection = float3(0,0,1);
static const float  kSunStrength = 1.25;
static const float  kCloudDensity = 2000.0;
static const float  kSurfaceRotation = 0.1;

struct VertexOutput
{
    float4 pos : SV_Position;
    float3 world_pos : TEXCOORD1;
};

Texture2D<float4> planet_albedo;
Texture2D<float>  planet_height;
Texture2D<float4> planet_clouds;
SamplerState sampler;

uniform float4x4 view_projection;
uniform float3 camera_position;
uniform float cloud_rotation;
uniform uint sphere_resolution;

float sqr(float a)
{
    return a * a;
}

bool intersect_sphere(float3 origin, float3 direction, float radius, out float2 t)
{
    // precise version
    float r2 = radius * radius;

    float DD = dot(direction, direction);
    float Df = dot(origin, direction);
    
    float DD2 = DD*DD;

    float3 l = DD * origin - direction*Df;
    float a2_r2 = DD2*r2;
    
    float ll = dot(l,l);

    if (a2_r2 < ll)
    {
        t = -1;
        return false;
    }
    
    float det = a2_r2 - ll;
    
    float rcp_a = 1.0 / DD;
    
    det = sqrt(det * rcp_a);
    
    t = (-Df + float2(-det, det)) * rcp_a;
    return t.y > 0;
}

float3 uv_to_xyz(float2 uv) {
    return float3(
        sin(uv.y * M_PI) * cos(uv.x * M_2PI),
        cos(uv.y * M_PI),
        sin(uv.y * M_PI) * sin(uv.x * M_2PI)
    );
}
float2 xyz_to_uv(float3 xyz)
{
    return float2(
        -atan2(xyz.z, -xyz.x) * M_1_2PI + 0.5,
        acos(xyz.y) * M_1_PI
    );
}

float3 get_surface_pos(float2 uv)
{
    return uv_to_xyz(uv) * (1.0 + kTerrainAltitude * planet_height.Sample(sampler, uv + float2(kSurfaceRotation, 0)));
}

float sample_cloud_density_n(float3 p)
{
    return kCloudDensity * planet_clouds.Sample(sampler, xyz_to_uv(p) + float2(0.01 * cloud_rotation, 0)).a;
}
float sample_cloud_density(float3 p)
{
    return sample_cloud_density_n(normalize(p));
}

float integrate_cloud_density<let kSteps : uint>(float3 ray_origin, float3 ray_direction, float t_max)
{
    float2 outer_t_range;
    if (!intersect_sphere(ray_origin, ray_direction, 1.0 + kCloudAltitudeRange.y, outer_t_range))
        return 0;
    if (outer_t_range.y < 0)
        return 0; // missed the entire atmosphere

    float2 t_range = float2(max(outer_t_range.x, 0), outer_t_range.y);

    float2 inner_t_range;
    if (intersect_sphere(ray_origin, ray_direction, 1.0 + kCloudAltitudeRange.x, inner_t_range))
    {
        if (inner_t_range.x > 0)
            t_range.y = inner_t_range.x;
        else
            t_range.x = inner_t_range.y;
    }

    t_range.y = min(t_range.y, t_max);
    if (t_range.y <= t_range.x)
        return 0;
    
    float total_density = 0;

    float t_step = (t_range.y - t_range.x) / kSteps;
    for (uint i = 0; i < kSteps; i++)
    {
        float t = lerp(t_range.x, t_range.y, float(i + 0.5) / kSteps);
        total_density += t_step * sample_cloud_density(ray_origin + ray_direction * t);
    }

    if (inner_t_range.x > 0)
    {
        t_range = float2(inner_t_range.y, min(outer_t_range.y, t_max));
        if (t_range.x < t_range.y)
        {
            t_step = (t_range.y - t_range.x) / kSteps;
            for (uint i = 0; i < kSteps; i++)
            {
                float t = lerp(t_range.x, t_range.y, float(i + 0.5) / kSteps);
                total_density += t_step * sample_cloud_density(ray_origin + ray_direction * t);
            }
        }
    }

    return total_density;
}

float3 eval_cloud_phase(float3 dir_in, float3 dir_out)
{
    return 1;
    // float g = 0; // g^2
    // return ((1 - sqr(g)) / pow(1 + sqr(g) - 2*g*dot(-dir_in, dir_out), 1.5)) / M_4PI;
}
float4 integrate_cloud_color<let kSteps : uint, let kShadowSteps : uint>(float3 ray_origin, float3 ray_direction, float t_max)
{
    float2 outer_t_range;
    if (!intersect_sphere(ray_origin, ray_direction, 1.0 + kCloudAltitudeRange.y, outer_t_range))
        return 0;
    if (outer_t_range.y < 0)
        return 0; // missed the entire atmosphere

    float2 t_range = float2(max(outer_t_range.x, 0), outer_t_range.y);

    float2 inner_t_range;
    if (intersect_sphere(ray_origin, ray_direction, 1.0 + kCloudAltitudeRange.x, inner_t_range))
    {
        if (inner_t_range.x > 0)
            t_range.y = inner_t_range.x;
        else
            t_range.x = inner_t_range.y;
    }

    t_range.y = min(t_range.y, t_max);
    if (t_range.y <= t_range.x)
        return 0;

    float total_density = 0;
    float3 total_color = 0;

    float t_step = (t_range.y - t_range.x) / kSteps;
    for (uint i = 0; i < kSteps; i++)
    {
        float t = lerp(t_range.x, t_range.y, float(i + 0.5) / kSteps);
        float3 p = ray_origin + ray_direction * t;
        float d = sample_cloud_density(p);

        float2 surface_t;
        bool hit = intersect_sphere(p, kSunDirection, 1.0, surface_t);
        if (!hit || surface_t.y < 0)
        {
            total_color += exp(-total_density) * (t_step * d * eval_cloud_phase(-ray_direction, kSunDirection)) * exp(-integrate_cloud_density<kShadowSteps>(p, kSunDirection, 1e9)) * kSunStrength;
        }

        total_density += t_step * d;
    }

    if (inner_t_range.x > 0)
    {
        t_range = float2(inner_t_range.y, min(outer_t_range.y, t_max));
        if (t_range.x < t_range.y)
        {
            t_step = (t_range.y - t_range.x) / kSteps;
            for (uint i = 0; i < kSteps; i++)
            {
                float t = lerp(t_range.x, t_range.y, float(i + 0.5) / kSteps);
                float3 p = ray_origin + ray_direction * t;
                float d = sample_cloud_density(p);

                float2 surface_t;
                bool hit = intersect_sphere(p, kSunDirection, 1.0, surface_t);
                if (!hit || surface_t.y < 0)
                {
                    total_color += exp(-total_density) * (t_step * d * eval_cloud_phase(-ray_direction, kSunDirection)) * exp(-integrate_cloud_density<kShadowSteps>(p, kSunDirection, 1e9)) * kSunStrength;
                }

                total_density += t_step * d;
            }
        }
    }

    return float4(total_color, 1 - exp(-total_density));
}


[shader("vertex")]
VertexOutput vs(uint vertexId : SV_VertexID)
{
    uint patchId = vertexId / 6;

    static const uint2 offsets[] = {
        uint2(0, 0), uint2(0, 1), uint2(1, 0),
        uint2(1, 0), uint2(0, 1), uint2(1, 1)
    };

    uint2 patchCoord = uint2(patchId % sphere_resolution, patchId / sphere_resolution);
    uint2 ij = patchCoord + offsets[vertexId % 6];

    float2 uv = ij / float(sphere_resolution);

    float3 pos = uv_to_xyz(uv) * (1 + kCloudAltitudeRange.y);

    VertexOutput v;
    v.pos = mul(view_projection, float4(pos, 1));
    v.world_pos = pos;
    return v;
}

[shader("fragment")]
float4 fs(VertexOutput v) : SV_Target
{
    float3 ray_direction = normalize(v.world_pos - camera_position);

    float4 color = 0;

    float2 surface_t;
    if (intersect_sphere(camera_position, ray_direction, 1.0, surface_t))
    {
        if (surface_t.x < 0 && surface_t.y > 0)
            return float4(0, 0, 0, 1); // below surface?

        if (surface_t.x > 0)
        {
            float3 p = normalize(camera_position + ray_direction * surface_t.x);
            float2 uv = xyz_to_uv(p);

            uint2 dims;
            planet_height.GetDimensions(dims.x, dims.y);
            float3 p0  = get_surface_pos(uv);
            float3 p10 = get_surface_pos(uv + float2(1.0 / dims.x, 0));
            float3 p01 = get_surface_pos(uv + float2(0, 1.0 / dims.y));
            float3 tangent   = normalize(p10 - p0);
            float3 bitangent = normalize(p01 - p0);
            bitangent = normalize(bitangent - tangent * dot(tangent, bitangent));
            float3 normal = normalize(cross(tangent, bitangent));

            float3 albedo = planet_albedo.Sample(sampler, uv + float2(kSurfaceRotation, 0)).rgb;

            float3 shadow_ray_origin = p * (1 + kCloudAltitudeRange.x*0.5);
            float2 shadow_t_range;
            bool hit = intersect_sphere(shadow_ray_origin, kSunDirection, 1.0, shadow_t_range);
            if (!hit || shadow_t_range.y > 0)
            {
                // lambertian
                float3 brdf = albedo * max(0, dot(kSunDirection, normal));

                float shadow_density = integrate_cloud_density<50>(shadow_ray_origin, kSunDirection, 1e9);
                color = float4(brdf * kSunStrength * exp(-shadow_density), 1);
            }
        }
    }
    else
    {
        surface_t.x = 1e9;
    }

    float4 cloud_color = integrate_cloud_color<50, 50>(camera_position, ray_direction, surface_t.x);

    // composite clouds over land
    color.rgb = lerp(color.rgb, cloud_color.rgb, cloud_color.a);
    color.a = saturate(color.a + cloud_color.a);
    return color;
}