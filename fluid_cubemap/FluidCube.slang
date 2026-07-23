import Math;

struct CubeMapTangentFrame
{
    float3 n, t, b;

    __init(float3 dir)
    {
        n = dir;
        float3 uvw = dir_to_cube_uvw(dir);
        let dx = __fwd_diff(cube_uvw_to_dir)(diffPair(uvw, float3(1, 0, 0)));
        let dy = __fwd_diff(cube_uvw_to_dir)(diffPair(uvw, float3(0, 1, 0)));

        t = normalize(dx.d);
        
        // orthonormalize b against t and dir
        b = dy.d;
        b -= dir * dot(b, dir);
        b -=  t * dot(b,  t);
        b = normalize(b);
    }

    float2 to_local(float3 vec)
    {
        float l = length(vec);
        vec -= n * dot(vec, n);
        vec = safe_normalize(vec) * l;
        return float2(dot(vec, t), dot(vec, b));
    }

    float3 to_world(float2 vec)
    {
        return t * vec.x + b * vec.y;
    }
}

uint2 cube_to_2d(uint3 texel, uint resolution)
{
    return uint2(texel.xy + uint2(texel.z * resolution, 0));
}

// texel index of `cell` offset by one texel along the face's u/v axes,
// stepping onto the neighboring face (with the correct rotated uv) when
// the offset crosses a face edge
uint3 cube_neighbor_texel(uint3 cell, int2 offset, uint resolution)
{
    int2 n = int2(cell.xy) + offset;
    if (all(n >= 0) && all(n < int(resolution)))
        return uint3(n, cell.z);

    // extend this face's uv plane past its [0,1) range; cube_uvw_to_dir's
    // per-face mapping is just a plane parameterization, so it stays valid
    // beyond the face bounds, and dir_to_cube_uvw then reclassifies the
    // resulting direction onto the correct neighboring face
    float3 dir = cube_uvw_to_dir(float3((float2(cell.xy) + 0.5 + offset) / float(resolution), cell.z));
    float3 uvw = dir_to_cube_uvw(dir);
    uint2 wrapped = uint2(clamp(int2(uvw.xy * resolution), 0, int(resolution) - 1));
    return uint3(wrapped, uint(uvw.z));
}

float sample_cube(Texture2D<float> cubemap, float3 texel, uint mip)
{
    uint w, resolution, mips;
    cubemap.GetDimensions(0, w, resolution, mips);
    resolution >>= mip;

    // home texel (same as nearest-sample lookup) plus the fractional
    // offset of `uvw` from its center, in texel units
    int2 home = clamp(int2(texel.xy * resolution), 0, int(resolution) - 1);
    float2 frac_uv = texel.xy * resolution - float2(home);

    // the four texels to blend are `home` and its neighbors toward
    // whichever side of the texel center `texel` falls on; `off` picks
    // that corner (-1 or 0 per axis) and `frac` is the blend weight
    // relative to it
    int2 off = int2(frac_uv.x < 0.5 ? -1 : 0, frac_uv.y < 0.5 ? -1 : 0);
    float2 frac = frac_uv - 0.5 - float2(off);

    uint3 home_cell = uint3(uint2(home), uint(texel.z));
    float s00 = cubemap.Load(int3(cube_to_2d( cube_neighbor_texel(home_cell, off, resolution), resolution ), mip));
    float s10 = cubemap.Load(int3(cube_to_2d( cube_neighbor_texel(home_cell, off + int2(1, 0), resolution), resolution ), mip));
    float s01 = cubemap.Load(int3(cube_to_2d( cube_neighbor_texel(home_cell, off + int2(0, 1), resolution), resolution ), mip));
    float s11 = cubemap.Load(int3(cube_to_2d( cube_neighbor_texel(home_cell, off + int2(1, 1), resolution), resolution ), mip));

    return lerp(lerp(s00, s10, frac.x), lerp(s01, s11, frac.x), frac.y);
}

struct FluidCube
{
    // tangent-space velocity
    Texture2D<float2>   _velocity;
    RWTexture2D<float2> _velocity_rw;

    Texture2D<float>   _smoke;
    RWTexture2D<float> _smoke_rw;

    uint resolution;

    float2 load_velocity(uint3 texel)
    {
        return _velocity[cube_to_2d(texel, resolution)];
    }
    float2 load_velocity_rw(uint3 texel)
    {
        return _velocity_rw[cube_to_2d(texel, resolution)];
    }
    void store_velocity(uint3 texel, float2 newValue)
    {
        _velocity_rw[cube_to_2d(texel, resolution)] = newValue;
    }

    float load_smoke(uint3 texel)
    {
        return _smoke[cube_to_2d(texel, resolution)];
    }
    float load_smoke_rw(uint3 texel)
    {
        return _smoke_rw[cube_to_2d(texel, resolution)];
    }
    void store_smoke(uint3 texel, float newValue)
    {
        _smoke_rw[cube_to_2d(texel, resolution)] = newValue;
    }

    // assumes dir is normalized
    float sample_smoke(float3 dir)
    {
        return sample_cube(_smoke, dir_to_cube_uvw(dir), 0);
    }

    float3 load_world_space_velocity(uint3 texel)
    {
        float3 dir = cube_uvw_to_dir(float3((texel.xy + 0.5) / resolution, texel.z));
        return CubeMapTangentFrame(dir).to_world(load_velocity(texel));
    }
    float3 load_world_space_velocity_rw(uint3 texel)
    {
        float3 dir = cube_uvw_to_dir(float3((texel.xy + 0.5) / resolution, texel.z));
        return CubeMapTangentFrame(dir).to_world(load_velocity_rw(texel));
    }

    // assumes dir is normalized
    float3 sample_velocity(float3 dir)
    {
        float3 uvw = dir_to_cube_uvw(dir);

        // home texel (same as nearest-sample lookup) plus the fractional
        // offset of `uvw` from its center, in texel units
        int2 home = clamp(int2(uvw.xy * resolution), 0, int(resolution) - 1);
        float2 frac_uv = uvw.xy * resolution - float2(home);

        // the four texels to blend are `home` and its neighbors toward
        // whichever side of the texel center `uvw` falls on; `off` picks
        // that corner (-1 or 0 per axis) and `frac` is the blend weight
        // relative to it
        int2 off = int2(frac_uv.x < 0.5 ? -1 : 0, frac_uv.y < 0.5 ? -1 : 0);
        float2 frac = frac_uv - 0.5 - float2(off);

        uint3 home_cell = uint3(uint2(home), uint(uvw.z));
        float3 s00 = load_world_space_velocity(cube_neighbor_texel(home_cell, off, resolution));
        float3 s10 = load_world_space_velocity(cube_neighbor_texel(home_cell, off + int2(1, 0), resolution));
        float3 s01 = load_world_space_velocity(cube_neighbor_texel(home_cell, off + int2(0, 1), resolution));
        float3 s11 = load_world_space_velocity(cube_neighbor_texel(home_cell, off + int2(1, 1), resolution));

        return lerp(lerp(s00, s10, frac.x), lerp(s01, s11, frac.x), frac.y);
    }
};
