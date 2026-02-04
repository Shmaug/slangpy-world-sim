import Math;
import RNG;

static const float  kEarthRadius = 6371; // kilometers
static const float  kTerrainAltitude = 6.4 / kEarthRadius;
static const float2 kCloudAltitudeRange = float2(2, 10) / kEarthRadius;
static const float3 kSunDirection = float3(0,0,1);

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
uniform float3   camera_position;
uniform uint     sphere_resolution;
uniform float3   sun_emission;
uniform float    cloud_density;
uniform float    surface_rotation;
uniform float    cloud_rotation;
uniform float    atmosphere_rayleigh_height;
uniform float    atmosphere_mie_height;
uniform float3   atmosphere_rayleigh_color;
uniform float    atmosphere_mie_color;
uniform float    atmosphere_height;
uniform float    atmosphere_density;
uniform uint     frame_seed;

float3 get_surface_pos(float2 uv)
{
    return uv_to_xyz(uv) * (1.0 + kTerrainAltitude * planet_height.Sample(sampler, uv + float2(surface_rotation, 0)));
}

void sample_atmosphere_density(float3 p, out float3 rayleigh, out float mie, out float cloud)
{
    float len_p = length(p);
    float height = len_p - 1.0;

    if (cloud_density > 0 && height >= kCloudAltitudeRange.x && height <= kCloudAltitudeRange.y)
        cloud = cloud_density * planet_clouds.Sample(sampler, xyz_to_uv(p / len_p) + float2(cloud_rotation, 0)).a;
    else
        cloud = 0;
    
    rayleigh = atmosphere_density * atmosphere_rayleigh_color * exp(-height * atmosphere_rayleigh_height);
    mie      = atmosphere_density * atmosphere_mie_color * exp(-height * atmosphere_mie_height);
}

// Integrate density along a ray. Returns the per-channel optical depth.
float3 integrate_atmosphere_density<let kSteps : uint>(float3 ray_origin, float3 ray_direction, inout RNG rng)
{
    float2 outer_t_range;
    if (!intersect_sphere(ray_origin, ray_direction, 1.0 + kCloudAltitudeRange.y, outer_t_range))
        return 0;
    if (outer_t_range.y < 0)
        return 0; // missed the entire atmosphere

    float2 t_range = float2(max(outer_t_range.x, 0), outer_t_range.y);
    if (t_range.y <= t_range.x)
        return 0;

    float3 optical_depth = 0;
    float t_offset = rng.nextFloat() * 1.0;
    float t_step = (t_range.y - t_range.x) / kSteps;
    for (uint i = 0; i < kSteps; i++)
    {
        float t = lerp(t_range.x, t_range.y, float(i + t_offset) / kSteps);
        float3 r;
        float m, c;
        sample_atmosphere_density(ray_origin + ray_direction * t, r, m, c);
        optical_depth += t_step * (r + m + c);
    }
    return optical_depth;
}

// cos_theta = dot(-dir_in, dir_out)
float eval_phase_henyey_greenstein(float cos_theta, float g)
{
    float g2 = sqr(g);
    return ((3 * (1 - g2)) / (2 * (2 + g2))) * ((1 + sqr(cos_theta)) / pow(1 + g2 - 2 * g * cos_theta, 1.5));
}

void integrate_atmosphere_color<let kSteps : uint, let kShadowSteps : uint>(float3 ray_origin, float3 ray_direction, float t_max, inout RNG rng, out float3 inscatter, out float3 transmittance)
{
    inscatter = 0;
    transmittance = 1;

    float2 outer_t_range;
    if (!intersect_sphere(ray_origin, ray_direction, 1.0 + atmosphere_height, outer_t_range))
        return;
    if (outer_t_range.y < 0)
        return; // missed the entire atmosphere

    float2 t_range = float2(max(outer_t_range.x, 0), min(outer_t_range.y, t_max));
    if (t_range.y <= t_range.x)
        return;

    float mu = dot(ray_direction, kSunDirection);
    float phase_r = eval_phase_henyey_greenstein(mu, 0.0);
    float phase_m = eval_phase_henyey_greenstein(mu, 0.76);
    float phase_c = eval_phase_henyey_greenstein(mu, 0.0);
    float t_step = (t_range.y - t_range.x) / kSteps;

    float3 optical_depth = 0;

    float t_offset = rng.nextFloat() * 1.0;
    for (uint i = 0; i < kSteps; i++)
    {
        float3 p = ray_origin + ray_direction * lerp(t_range.x, t_range.y, float(i + t_offset) / kSteps);
        float height = length(p) - 1.0;
        if (height <= 0)
            break;

        // atmosphere density at current height
        float3 dr;
        float dm, dc;
        sample_atmosphere_density(p, dr, dm, dc);
        dr *= t_step;
        dm *= t_step;
        dc *= t_step;

        float2 surface_t;
        bool hit = intersect_sphere(p, kSunDirection, 1.0, surface_t);
        if (!hit || surface_t.y < 0)
        {
            float2 shadow_t_range;
            if (intersect_sphere(p, kSunDirection, 1.0 + atmosphere_height, shadow_t_range))
            {
                float3 shadow_optical_depth = integrate_atmosphere_density<kShadowSteps>(p, kSunDirection, rng);
                float3 shadow_transmittance = exp(-(optical_depth + shadow_optical_depth));
                inscatter += shadow_transmittance * (dr * phase_r + dm * phase_m + dc * phase_c);
            }
        }

        optical_depth += dr + dm + dc;
    }

    inscatter *= sun_emission;
    transmittance = exp(-optical_depth);
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

    float3 pos = uv_to_xyz(uv) * (1 + atmosphere_height*1.5);

    VertexOutput v;
    v.pos = mul(view_projection, float4(pos, 1));
    v.world_pos = pos;
    return v;
}

[shader("fragment")]
float4 fs(VertexOutput v) : SV_Target
{
    RNG rng = RNG(pcg(frame_seed + pcg(int(v.pos.y) + pcg(int(v.pos.x)))));

    float3 ray_direction = normalize(v.world_pos - camera_position);
    
    // note: alpha channel stores transmittance, not alpha. transmittance=1-alpha
    float4 color = float4(0, 0, 0, 1);

    float2 surface_t;
    if (intersect_sphere(camera_position, ray_direction, 1.0, surface_t))
    {
        if (surface_t.x > 0)
        {
            float3 p = normalize(camera_position + ray_direction * surface_t.x);
            float2 uv = xyz_to_uv(p);

            /*
            float3 normal = p;
            /*/
            uint2 dims;
            planet_height.GetDimensions(dims.x, dims.y);
            float3 p0  = get_surface_pos(uv);
            float3 p10 = get_surface_pos(uv + float2(1.0 / dims.x, 0));
            float3 p01 = get_surface_pos(uv + float2(0, 1.0 / dims.y));
            float3 tangent   = normalize(p10 - p0);
            float3 bitangent = normalize(p01 - p0);
            bitangent = normalize(bitangent - tangent * dot(tangent, bitangent));
            float3 normal = normalize(cross(bitangent, tangent));
            // */

            float3 albedo = planet_albedo.Sample(sampler, uv + float2(surface_rotation, 0)).rgb;

            float3 shadow_ray_origin = p * (1 + kTerrainAltitude);
            
            // lambertian
            float3 brdf = albedo * max(0, dot(kSunDirection, normal));

            float3 shadow_density = integrate_atmosphere_density<50>(shadow_ray_origin, kSunDirection, rng);
            color = float4(brdf * sun_emission * exp(-shadow_density), 0);
        }
        else
        {
            if (surface_t.y > 0)
                return 0; // below surface?
            surface_t.x = 1e9; // planet behind ray
        }
    }
    else
    {
        surface_t.x = 1e9;
    }

    float3 inscatter;
    float3 transmittance;
    integrate_atmosphere_color<100, 20>(camera_position, ray_direction, surface_t.x, rng, inscatter, transmittance);
    color.rgb = transmittance * color.rgb + inscatter;
    color.a *= luminance(transmittance);
    return color;
}