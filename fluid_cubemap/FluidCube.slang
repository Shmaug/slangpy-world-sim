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

struct FluidCube
{
    // tangent-space velocity
    Texture2DArray<float2>    velocity;
    RWTexture2DArray<float2>  velocity_rw;

    Texture2DArray<float>    smoke;
    RWTexture2DArray<float>  smoke_rw;

    uint resolution;

    // texel index of `cell` offset by one texel along the face's u/v axes,
    // stepping onto the neighboring face (with the correct rotated uv) when
    // the offset crosses a face edge
    uint3 neighbor_cell(uint3 cell, int2 offset)
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

    // assumes dir is normalized
    float sample_smoke(float3 dir)
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
        float s00 = smoke[neighbor_cell(home_cell, off)];
        float s10 = smoke[neighbor_cell(home_cell, off + int2(1, 0))];
        float s01 = smoke[neighbor_cell(home_cell, off + int2(0, 1))];
        float s11 = smoke[neighbor_cell(home_cell, off + int2(1, 1))];

        return lerp(lerp(s00, s10, frac.x), lerp(s01, s11, frac.x), frac.y);
    }

    float3 fetch_velocity(uint3 texel)
    {
        float3 dir = cube_uvw_to_dir(float3((texel.xy + 0.5) / resolution, texel.z));
        return CubeMapTangentFrame(dir).to_world(velocity[texel]);
    }
    float3 fetch_velocity_rw(uint3 texel)
    {
        float3 dir = cube_uvw_to_dir(float3((texel.xy + 0.5) / resolution, texel.z));
        return CubeMapTangentFrame(dir).to_world(velocity_rw[texel]);
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
        float3 s00 = fetch_velocity(neighbor_cell(home_cell, off));
        float3 s10 = fetch_velocity(neighbor_cell(home_cell, off + int2(1, 0)));
        float3 s01 = fetch_velocity(neighbor_cell(home_cell, off + int2(0, 1)));
        float3 s11 = fetch_velocity(neighbor_cell(home_cell, off + int2(1, 1)));

        return lerp(lerp(s00, s10, frac.x), lerp(s01, s11, frac.x), frac.y);
    }
};
