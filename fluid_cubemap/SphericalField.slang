import Math;

// great-circle (geodesic) distance in radians between the centers of two texels
[Differentiable]
float sphere_geodesic_distance(float3 a, float3 b)
{
    return acos(clamp(dot(a, b), -1.0, 1.0));
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
    int face = int(uvw.z) % 6;

    float3 dir;
    if      (face == 0) dir = float3( 1.0,  sc.y, -sc.x);
    else if (face == 1) dir = float3(-1.0,  sc.y,  sc.x);
    else if (face == 2) dir = float3(sc.x,  1.0,  -sc.y);
    else if (face == 3) dir = float3(sc.x, -1.0,   sc.y);
    else if (face == 4) dir = float3(sc.x,  sc.y,   1.0);
    else                dir = float3(-sc.x, sc.y,  -1.0);

    return normalize(dir);
}

uint flatten_cube_texel(uint3 texel, uint resolution, uint vertical_resolution, uint mip = 0)
{
    uint offset = 0;
    for (uint i = 0; i < mip; i++)
        offset += (resolution >> i) * (resolution >> i) * 6 * vertical_resolution;
    uint mip_res = resolution >> mip;
    return offset + morton2d(texel.xy) + texel.z * mip_res * mip_res;
}

// steps `cell` one layer up/down (same face); returns false at a closed
// boundary (layer would go out of [0, vertical_resolution)), in which case
// `neighbor` is set to `cell` itself and the caller must apply the
// solid-wall ghost value appropriate to the field (sign-flip for the
// wall-normal velocity component, same-value copy for a Neumann scalar
// like pressure/divergence)
bool vertical_neighbor(uint3 cell, int dz, uint vertical_resolution, out uint3 neighbor)
{
    int layer = int(cell.z / 6) + dz;
    if (layer < 0 || layer >= int(vertical_resolution))
    {
        neighbor = cell;
        return false;
    }
    neighbor = uint3(cell.xy, cell.z % 6 + 6 * uint(layer));
    return true;
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
    return uint3(wrapped, uint(uvw.z) + 6*(cell.z/6));
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
        vec -= n * dot(vec, n); // remove normal component
        return float2(dot(vec, t), dot(vec, b));
    }

    // as to_local, but keeps the normal component as .z instead of
    // discarding it -- used to round-trip a full (tangential + radial)
    // world-space vector into local storage
    float3 to_local3(float3 vec)
    {
        return float3(dot(vec, t), dot(vec, b), dot(vec, n));
    }

    float3 to_world(float3 vec)
    {
        return t * vec.x + b * vec.y + n * vec.z;
    }
}

struct SphericalScalarField
{
    RWStructuredBuffer<float> data;
    float thickness; // height above 1 of the simulation
    uint resolution;
    uint vertical_resolution;

    uint flatten_index(uint3 texel, uint mip = 0)
    {
        return flatten_cube_texel(texel, resolution, vertical_resolution, mip);
    }

    bool bounds_check(int3 texel)
    {
        return all(texel >= 0) && all(texel.xy < resolution) && texel.z < 6 * vertical_resolution;
    }

    float load(int3 texel, uint mip = 0)
    {
        return bounds_check(texel) ? data[flatten_index(texel, mip)] : 0;
    }
    void store(int3 texel, float value, uint mip = 0)
    {
        if (bounds_check(texel))
            data[flatten_index(texel, mip)] = value;
    }

    uint3 get_neighbor(uint3 texel, int2 offset, uint mip = 0)
    {
        return cube_neighbor_texel(texel, offset, resolution >> mip);
    }

    float sample_at_layer(float3 dir, uint mip, uint vertical_layer)
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

        uint3 home_cell = uint3(uint2(home), uint(cube_uvw.z) + 6 * vertical_layer);
        float s00 = load(get_neighbor(home_cell, off + int2(0, 0), mip), mip);
        float s10 = load(get_neighbor(home_cell, off + int2(1, 0), mip), mip);
        float s01 = load(get_neighbor(home_cell, off + int2(0, 1), mip), mip);
        float s11 = load(get_neighbor(home_cell, off + int2(1, 1), mip), mip);

        return lerp(lerp(s00, s10, frac.x), lerp(s01, s11, frac.x), frac.y);
    }

    // as sample_at_layer, but blends between the two integer layers
    // straddling `vertical_layer` -- lets a fractional radial position
    // (as produced by 3D advection) sample smoothly across layers
    float sample(float3 dir, uint mip = 0, float vertical_layer = 0)
    {
        float layer_clamped = clamp(vertical_layer, 0, float(vertical_resolution - 1));
        uint layer_lo = uint(floor(layer_clamped));
        uint layer_hi = min(layer_lo + 1, vertical_resolution - 1);
        float t = layer_clamped - float(layer_lo);
        return lerp(sample_at_layer(dir, mip, layer_lo), sample_at_layer(dir, mip, layer_hi), t);
    }
}

struct SphericalVelocityField
{
    // local-frame velocity field: .xy is tangential (t,b), .z is the radial
    // (wall-normal) component, in world units per time (arc length = world
    // distance since the sphere has radius 1)
    RWStructuredBuffer<float3> data;
    float thickness; // height above 1 of the simulation
    uint resolution;
    uint vertical_resolution;

    uint flatten_index(uint3 texel, uint mip = 0)
    {
        return flatten_cube_texel(texel, resolution, vertical_resolution, mip);
    }

    bool bounds_check(int3 texel)
    {
        return all(texel >= 0) && all(texel.xy < resolution) && texel.z < vertical_resolution * 6;
    }

    float3 load(int3 texel, uint mip = 0)
    {
        return bounds_check(texel) ? data[flatten_index(texel, mip)] : 0;
    }
    void store(int3 texel, float3 value, uint mip = 0)
    {
        if (bounds_check(texel))
            data[flatten_index(texel, mip)] = value;
    }

    uint3 get_neighbor(uint3 texel, int2 offset)
    {
        return cube_neighbor_texel(texel, offset, resolution);
    }

    float3 to_local(float3 pos, float3 vel)
    {
        return CubeMapTangentFrame(pos).to_local3(vel);
    }

    float3 load_world_space(uint3 texel)
    {
        float3 dir = cube_uvw_to_dir(float3((texel.xy + 0.5) / resolution, texel.z));
        return CubeMapTangentFrame(dir).to_world(load(texel));
    }
    float3 load_world_space(uint3 texel, float3 dst_dir)
    {
        float3 dir = cube_uvw_to_dir(float3((texel.xy + 0.5) / resolution, texel.z));
        float3 vel = CubeMapTangentFrame(dir).to_world(load(texel));
        return sphere_parallel_transport(dir, dst_dir, vel);
    }

    float3 sample_at_layer(float3 dir, uint vertical_layer)
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

        uint3 home_cell = uint3(uint2(home), uint(cube_uvw.z) + 6 * vertical_layer);

        float3 s00 = load_world_space(get_neighbor(home_cell, off + int2(0, 0)), dir);
        float3 s10 = load_world_space(get_neighbor(home_cell, off + int2(1, 0)), dir);
        float3 s01 = load_world_space(get_neighbor(home_cell, off + int2(0, 1)), dir);
        float3 s11 = load_world_space(get_neighbor(home_cell, off + int2(1, 1)), dir);

        return lerp(lerp(s00, s10, frac.x), lerp(s01, s11, frac.x), frac.y);
    }

    // as sample_at_layer, but blends between the two integer layers
    // straddling `vertical_layer`; no extra transport is needed between
    // them since both layers share the same (dir) angular footprint and
    // therefore the same tangent frame
    float3 sample(float3 dir, float vertical_layer = 0)
    {
        float layer_clamped = clamp(vertical_layer, 0, float(vertical_resolution - 1));
        uint layer_lo = uint(floor(layer_clamped));
        uint layer_hi = min(layer_lo + 1, vertical_resolution - 1);
        float t = layer_clamped - float(layer_lo);
        return lerp(sample_at_layer(dir, layer_lo), sample_at_layer(dir, layer_hi), t);
    }
}
