import Math;

// great-circle (geodesic) distance in radians between the centers of two texels
[Differentiable]
float sphere_geodesic_distance(float3 a, float3 b)
{
    return acos(clamp(dot(a, b), -1.0, 1.0));
}

// advances unit vector `pos` along the great circle in the direction of
// tangent vector `v` (tangent to the sphere at `pos`) by angle |v|
[Differentiable]
float3 sphere_exp_map(float3 pos, float3 v)
{
    float theta = length(v);
    if (theta < 1e-8) return pos;
    return pos * cos(theta) + (v / theta) * sin(theta);
}

// parallel-transports tangent vector `vec` (tangent to the sphere at
// `from`) along the great-circle arc from `from` to `to`, returning the
// corresponding tangent vector at `to`
[Differentiable]
float3 sphere_parallel_transport(float3 from, float3 to, float3 vec)
{
    float3 axis = cross(from, to);
    float s = length(axis);
    float c = dot(from, to);
    if (s < 1e-8) return vec;
    axis /= s;
    return vec * c + cross(axis, vec) * s + axis * dot(axis, vec) * (1.0 - c);
}

// cube face order: 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z
[Differentiable]
float3 dir_to_cube_uvw(float3 dir)
{
    float3 absDir = abs(dir);
    float maxAxis, uc, vc;
    int face;

    if (absDir.x >= absDir.y && absDir.x >= absDir.z)
    {
        maxAxis = absDir.x;
        if (dir.x > 0) { uc = -dir.z; vc = dir.y; face = 0; }
        else           { uc =  dir.z; vc = dir.y; face = 1; }
    }
    else if (absDir.y >= absDir.z)
    {
        maxAxis = absDir.y;
        if (dir.y > 0) { uc = dir.x; vc = -dir.z; face = 2; }
        else           { uc = dir.x; vc =  dir.z; face = 3; }
    }
    else
    {
        maxAxis = absDir.z;
        if (dir.z > 0) { uc =  dir.x; vc = dir.y; face = 4; }
        else           { uc = -dir.x; vc = dir.y; face = 5; }
    }

    float2 uv = float2(uc, vc) / maxAxis * 0.5 + 0.5;
    return float3(uv, face);
}

[Differentiable]
float3 cube_uvw_to_dir(float3 uvw)
{
    float2 sc = uvw.xy * 2.0 - 1.0;
    int face = int(uvw.z);

    float3 dir;
    if      (face == 0) dir = float3( 1.0,  sc.y, -sc.x);
    else if (face == 1) dir = float3(-1.0,  sc.y,  sc.x);
    else if (face == 2) dir = float3(sc.x,  1.0,  -sc.y);
    else if (face == 3) dir = float3(sc.x, -1.0,   sc.y);
    else if (face == 4) dir = float3(sc.x,  sc.y,   1.0);
    else                dir = float3(-sc.x, sc.y,  -1.0);

    return normalize(dir);
}

uint flatten_cube_texel(uint3 texel, uint resolution, uint mip = 0)
{
    uint offset = 0;
    for (uint i = 0; i < mip; i++)
        offset += (resolution >> i) * (resolution >> i) * 6;
    uint mip_res = resolution >> mip;
    return offset + morton2d(texel.xy) + texel.z * mip_res * mip_res;
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

struct SphericalScalarField
{
    RWStructuredBuffer<float> data;
    uint resolution;

    uint flatten_index(uint3 texel, uint mip = 0)
    {
        return flatten_cube_texel(texel, resolution, mip);
    }

    float load(uint3 texel, uint mip = 0)
    {
        return data[flatten_index(texel, mip)];
    }
    void store(uint3 texel, float value, uint mip = 0)
    {
        data[flatten_index(texel, mip)] = value;
    }

    uint3 get_neighbor(uint3 texel, int2 offset, uint mip = 0)
    {
        return cube_neighbor_texel(texel, offset, resolution >> mip);
    }

    float sample(float3 dir, uint mip = 0)
    {
        uint mip_res = resolution >> mip;
        float3 cube_uvw = dir_to_cube_uvw(dir);

        // home texel (same as nearest-sample lookup) plus the fractional
        // offset of `uvw` from its center, in texel units
        int2 home = clamp(int2(cube_uvw.xy * mip_res), 0, int(mip_res) - 1);
        float2 frac_uv = cube_uvw.xy * mip_res - float2(home);

        // the four texels to blend are `home` and its neighbors toward
        // whichever side of the texel center `texel` falls on; `off` picks
        // that corner (-1 or 0 per axis) and `frac` is the blend weight
        // relative to it
        int2 off = int2(frac_uv.x < 0.5 ? -1 : 0, frac_uv.y < 0.5 ? -1 : 0);
        float2 frac = frac_uv - 0.5 - float2(off);

        uint3 home_cell = uint3(uint2(home), uint(cube_uvw.z));
        float s00 = load(get_neighbor(home_cell, off + int2(0, 0), mip), mip);
        float s10 = load(get_neighbor(home_cell, off + int2(1, 0), mip), mip);
        float s01 = load(get_neighbor(home_cell, off + int2(0, 1), mip), mip);
        float s11 = load(get_neighbor(home_cell, off + int2(1, 1), mip), mip);

        return lerp(lerp(s00, s10, frac.x), lerp(s01, s11, frac.x), frac.y);
    }
}

struct SphericalVelocityField
{
    // tangent-space velocity field
    RWStructuredBuffer<float2> data;
    uint resolution;

    uint flatten_index(uint3 texel)
    {
        return flatten_cube_texel(texel, resolution);
    }

    float2 load(uint3 texel)
    {
        return data[flatten_index(texel)];
    }
    void store(uint3 texel, float2 value)
    {
        data[flatten_index(texel)] = value;
    }

    uint3 get_neighbor(uint3 texel, int2 offset)
    {
        return cube_neighbor_texel(texel, offset, resolution);
    }

    float3 load_world_space(uint3 texel)
    {
        float3 dir = cube_uvw_to_dir(float3((texel.xy + 0.5) / resolution, texel.z));
        return CubeMapTangentFrame(dir).to_world(data[flatten_index(texel)]);
    }
    float3 load_world_space(uint3 texel, float3 dst_dir)
    {
        float3 dir = cube_uvw_to_dir(float3((texel.xy + 0.5) / resolution, texel.z));
        float3 vel = CubeMapTangentFrame(dir).to_world(data[flatten_index(texel)]);
        return sphere_parallel_transport(dir, dst_dir, vel);
    }

    float3 sample(float3 dir)
    {
        float3 cube_uvw = dir_to_cube_uvw(dir);

        // home texel (same as nearest-sample lookup) plus the fractional
        // offset of `uvw` from its center, in texel units
        int2 home = clamp(int2(cube_uvw.xy * resolution), 0, int(resolution) - 1);
        float2 frac_uv = cube_uvw.xy * resolution - float2(home);

        // the four texels to blend are `home` and its neighbors toward
        // whichever side of the texel center `texel` falls on; `off` picks
        // that corner (-1 or 0 per axis) and `frac` is the blend weight
        // relative to it
        int2 off = int2(frac_uv.x < 0.5 ? -1 : 0, frac_uv.y < 0.5 ? -1 : 0);
        float2 frac = frac_uv - 0.5 - float2(off);

        uint3 home_cell = uint3(uint2(home), uint(cube_uvw.z));
        
        float3 s00 = load_world_space(get_neighbor(home_cell, off + int2(0, 0)), dir);
        float3 s10 = load_world_space(get_neighbor(home_cell, off + int2(1, 0)), dir);
        float3 s01 = load_world_space(get_neighbor(home_cell, off + int2(0, 1)), dir);
        float3 s11 = load_world_space(get_neighbor(home_cell, off + int2(1, 1)), dir);

        return lerp(lerp(s00, s10, frac.x), lerp(s01, s11, frac.x), frac.y);
    }
}
