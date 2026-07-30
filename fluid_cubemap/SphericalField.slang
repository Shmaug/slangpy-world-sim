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

struct CubeCoord
{
    int2 texel;
    int face;
    int layer;
    int channel;
}

uint flatten_cube_texel(CubeCoord coord, uint resolution, uint vertical_resolution, uint channels, uint mip = 0)
{
    uint mip_offset = 0;
    for (uint i = 0; i < mip; i++)
        mip_offset += (resolution >> i) * (resolution >> i) * 6 * vertical_resolution * channels;

    uint mip_res = resolution >> mip;
    return mip_offset + channels * ((mip_res * mip_res) * (6 * coord.layer + coord.face) + morton2d(coord.texel)) + coord.channel;
}

// steps `cell` one layer up/down (same face); returns false at a closed
// boundary (layer would go out of [0, vertical_resolution)), in which case
// `neighbor` is set to `cell` itself and the caller must apply the
// solid-wall ghost value appropriate to the field (sign-flip for the
// wall-normal velocity component, same-value copy for a Neumann scalar
// like pressure/divergence)
bool vertical_neighbor(CubeCoord cell, int dz, uint vertical_resolution, out CubeCoord neighbor)
{
    neighbor = cell;
    neighbor.layer = cell.layer + dz;
    return neighbor.layer >= 0 && neighbor.layer < int(vertical_resolution);
}

// texel index of `cell` offset by one texel along the face's u/v axes,
// stepping onto the neighboring face (with the correct rotated uv) when
// the offset crosses a face edge
CubeCoord cube_neighbor_texel(CubeCoord cell, int2 offset, uint resolution)
{
    CubeCoord neighbor = cell;
    neighbor.texel = int2(cell.texel) + offset;
    if (all(neighbor.texel >= 0) && all(neighbor.texel < int(resolution)))
    {
        return neighbor;
    }

    // extend this face's uv plane past its [0,1) range; cube_uvw_to_dir's
    // per-face mapping is just a plane parameterization, so it stays valid
    // beyond the face bounds, and dir_to_cube_uvw then reclassifies the
    // resulting direction onto the correct neighboring face
    float3 dir = cube_uvw_to_dir(float3((float2(cell.texel) + 0.5 + offset) / float(resolution), cell.face));
    float3 uvw = dir_to_cube_uvw(dir);
    neighbor.texel = int2(clamp(int2(uvw.xy * resolution), 0, int(resolution) - 1));
    neighbor.face = int(uvw.z);
    return neighbor;
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
    RWByteAddressBuffer data;
    float radius;
    float thickness; // height above 1 of the simulation
    float scale_height; // scale height where average density is found
    uint resolution;
    uint vertical_resolution;
    uint channels;

    // atmosphere density at physical radius `r` (in [1, 1+thickness]),
    // normalized to 1 at the surface r=1
    float sample_density(float r)
    {
        return scale_height > 0 ? exp(-(r - radius) / scale_height) : 1;
    }

    uint flatten_index(CubeCoord coord, uint mip = 0)
    {
        return flatten_cube_texel(coord, resolution, vertical_resolution, channels, mip);
    }

    bool bounds_check(CubeCoord coord, uint mip = 0)
    {
        return all(coord.texel >= 0) && all(coord.texel < resolution) && coord.face >= 0 && coord.face < 6 && coord.layer >= 0 && coord.layer < vertical_resolution && coord.channel < channels;
    }

    // requires N <= channels
    vector<float,N> load<int N>(CubeCoord coord, uint mip = 0)
    {
        return bounds_check(coord) ? data.Load<vector<float, N>>(sizeof(float) * flatten_index(coord, mip)) : 0;
    }
    // requires N <= channels
    void store<int N>(CubeCoord coord, vector<float, N> value, uint mip = 0)
    {
        if (bounds_check(coord))
            data.Store<vector<float, N>>(sizeof(float) * flatten_index(coord, mip), value);
    }

    CubeCoord get_neighbor(CubeCoord coord, int2 offset, uint mip = 0)
    {
        return cube_neighbor_texel(coord, offset, resolution >> mip);
    }

    vector<float,N> sample_layer<int N>(float3 dir, uint channel, uint vertical_layer, uint mip)
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

        CubeCoord home_cell = CubeCoord(uint2(home), uint(cube_uvw.z), vertical_layer, channel);
        let s00 = load<N>(get_neighbor(home_cell, off + int2(0, 0), mip), mip);
        let s10 = load<N>(get_neighbor(home_cell, off + int2(1, 0), mip), mip);
        let s01 = load<N>(get_neighbor(home_cell, off + int2(0, 1), mip), mip);
        let s11 = load<N>(get_neighbor(home_cell, off + int2(1, 1), mip), mip);

        return lerp(lerp(s00, s10, frac.x), lerp(s01, s11, frac.x), frac.y);
    }

    vector<float,N> sample<int N>(float3 dir, uint channel, float vertical_layer, uint mip)
    {
        int layer_lo = int(floor(vertical_layer));
        if (frac(vertical_layer) < 0.5)
            layer_lo -= 1;

        let z0 = sample_layer<N>(dir, channel, layer_lo, mip);
        let z1 = sample_layer<N>(dir, channel, layer_lo + 1, mip);

        return lerp(z0, z1, vertical_layer - float(layer_lo + 0.5));
    }

    // 3d sampling, vertical_layer extracted from length(pos); layers are
    // texel-centered (like a 3D texture), so the layer's r-center sits at
    // radius + thickness/vertical_resolution*(layer+0.5) -- hence the -0.5
    // to invert that back to a continuous layer coordinate
    vector<float,N> sample<int N>(float3 pos, uint channel, uint mip)
    {
        float h = length(pos);
        float3 dir = pos / h;
        float vertical_layer = vertical_resolution * ((h - radius) / thickness);
        return sample<N>(dir, channel, vertical_layer, mip);
    }

    vector<float,N> sample_layer_nearest<int N>(float3 dir, uint channel, int vertical_layer, uint mip)
    {
        float3 cube_uvw = dir_to_cube_uvw(dir);
        int2 texel = clamp(int2(cube_uvw.xy * (resolution >> mip)), 0, int(resolution >> mip) - 1);
        CubeCoord cell = CubeCoord(uint2(texel), uint(cube_uvw.z), vertical_layer, channel);
        return load<N>(cell);
    }

    vector<float,N> sample_nearest<int N>(float3 pos, uint channel, uint mip)
    {
        float h = length(pos);
        float3 dir = pos / h;
        float vertical_layer = vertical_resolution * ((h - radius) / thickness);
        return sample_layer_nearest<N>(dir, channel, int(floor(vertical_layer)), mip);
    }
}

struct SphericalVelocityField
{
    // local-frame velocity field: .xy is tangential (t,b), .z is the radial
    // (wall-normal) component, in world units per time (arc length = world
    // distance since the sphere has radius 1)
    RWByteAddressBuffer data;
    float radius;
    float thickness; // height above 1 of the simulation
    float scale_height; // scale height where average density is found
    uint resolution;
    uint vertical_resolution;

    // atmosphere density at physical radius `r` (in [1, 1+thickness]),
    // normalized to 1 at the surface r=1
    float sample_density(float r)
    {
        return scale_height > 0 ? exp(-(r - radius) / scale_height) : 1;
    }

    uint flatten_index(CubeCoord coord)
    {
        return flatten_cube_texel(coord, resolution, vertical_resolution, 1, 0);
    }

    bool bounds_check(CubeCoord coord)
    {
        return all(coord.texel >= 0) && all(coord.texel < resolution) && coord.face >= 0 && coord.face < 6 && coord.layer >= 0 && coord.layer < vertical_resolution && coord.channel == 0;
    }

    float3 load(CubeCoord coord)
    {
        return bounds_check(coord) ? data.Load<float3>(sizeof(float3) * flatten_index(coord)) : 0;
    }
    void store(CubeCoord coord, float3 value)
    {
        if (bounds_check(coord))
            data.Store<float3>(sizeof(float3) * flatten_index(coord), value);
    }

    CubeCoord get_neighbor(CubeCoord coord, int2 offset)
    {
        return cube_neighbor_texel(coord, offset, resolution);
    }

    float3 to_local(float3 dir, float3 vel)
    {
        return CubeMapTangentFrame(dir).to_local3(vel);
    }

    float3 load_world_space(CubeCoord coord)
    {
        float3 dir = cube_uvw_to_dir(float3((coord.texel + 0.5) / resolution, coord.face));
        return CubeMapTangentFrame(dir).to_world(load(coord));
    }
    float3 load_world_space(CubeCoord coord, float3 dst_dir)
    {
        float3 dir = cube_uvw_to_dir(float3((coord.texel + 0.5) / resolution, coord.face));
        float3 vel = CubeMapTangentFrame(dir).to_world(load(coord));
        return sphere_parallel_transport(dir, dst_dir, vel);
    }

    float3 sample_layer(float3 dir, int vertical_layer)
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

        CubeCoord home_cell = CubeCoord(uint2(home), uint(cube_uvw.z), vertical_layer, 0);

        float3 s00 = load_world_space(get_neighbor(home_cell, off + int2(0, 0)), dir);
        float3 s10 = load_world_space(get_neighbor(home_cell, off + int2(1, 0)), dir);
        float3 s01 = load_world_space(get_neighbor(home_cell, off + int2(0, 1)), dir);
        float3 s11 = load_world_space(get_neighbor(home_cell, off + int2(1, 1)), dir);

        return lerp(lerp(s00, s10, frac.x), lerp(s01, s11, frac.x), frac.y);
    }

    float3 sample(float3 dir, float vertical_layer)
    {
        int layer_lo = int(floor(vertical_layer));
        if (frac(vertical_layer) < 0.5)
            layer_lo -= 1;

        let z0 = sample_layer(dir, layer_lo);
        let z1 = sample_layer(dir, layer_lo + 1);

        return lerp(z0, z1, vertical_layer - float(layer_lo + 0.5));
    }

    // 3d sampling, vertical_layer extracted from length(pos)
    float3 sample(float3 pos)
    {
        float h = length(pos);
        float3 dir = pos / h;
        float vertical_layer = vertical_resolution * ((h - radius) / thickness);
        return sample(dir, vertical_layer);
    }

    float3 sample_layer_nearest(float3 dir, int vertical_layer)
    {
        float3 cube_uvw = dir_to_cube_uvw(dir);
        int2 texel = clamp(int2(cube_uvw.xy * resolution), 0, int(resolution) - 1);
        CubeCoord cell = CubeCoord(texel, uint(cube_uvw.z), vertical_layer, 0);
        return load_world_space(cell, dir);
    }

    float3 sample_nearest(float3 pos)
    {
        float h = length(pos);
        float3 dir = pos / h;
        float vertical_layer = vertical_resolution * ((h - radius) / thickness);
        return sample_layer_nearest(dir, int(floor(vertical_layer)));
    }
}
