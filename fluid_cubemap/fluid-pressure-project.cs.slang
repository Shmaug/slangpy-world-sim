import Math;
import SphericalField;

uniform SphericalVelocityField velocity;
uniform SphericalScalarField divergence;
uniform SphericalScalarField pressure_correction[2];
// mip level to read from for `resample`
uniform uint src_mip_level;
// mip level of pressure_correction (and divergence) that `step` relaxes this dispatch;
// `compute_divergence` and `apply` always operate at mip 0
uniform uint dst_mip_level;

Texture2D<float> surface_height;
Texture2D<float> surface_height2;
SamplerState linear_sampler;
uniform float terrain_height;

// mip level of `surface_height`/`surface_height2` whose texel footprint is
// roughly one fluid cell wide at `fluid_res`, so the sub-grid stats below
// are drawn from the same footprint the fluid solver actually sees
float heightmap_mip_level(uint fluid_res)
{
    uint2 dims;
    surface_height.GetDimensions(dims.x, dims.y);
    return max(0.0, log2(float(dims.x) / float(fluid_res)));
}

// mean and variance of terrain radius over the heightmap texels covered by
// one fluid cell's footprint at `mip` (surface_height2 holds height*height,
// so its mip chain gives E[h^2] where surface_height's gives E[h]^-shaped
// mean -- var = E[h^2] - E[h]^2)
void terrain_radius_stats(float3 dir, float mip, out float mean, out float variance)
{
    float2 uv = xyz_to_sphere_uv(dir);
    float h = surface_height.SampleLevel(linear_sampler, uv, mip);
    float h2 = surface_height2.SampleLevel(linear_sampler, uv, mip);
    mean = 1.0 + terrain_height * h;
    variance = sqr(terrain_height) * max(0.0, h2 - h * h);
}

// fraction of this cell's sub-grid footprint whose terrain sits above `r`,
// modeling the heightmap texels under the footprint as Gaussian(mean,
// variance) -- this is what lets a fluid cell straddling a sub-grid ridge
// register as partially blocked instead of snapping to fully open/solid
// depending only on whether the mean height crosses `r`. Degenerates to a
// hard step once the heightmap is no finer than the fluid grid (variance
// collapses to 0 at that mip).
float solid_fraction(float3 dir, float r, float mip)
{
    if (terrain_height > 0)
    {
        float mean, variance;
        terrain_radius_stats(dir, mip, mean, variance);
        if (variance < 1e-12)
            return r < mean ? 1.0 : 0.0;
        return saturate(0.5 * (1.0 - erf((r - mean) * M_SQRT1_2 / sqrt(variance))));
    }
    return 0;
}

// terrain only ever occupies the bottom-most fluid layer (layer 0, the one
// touching r=1) -- `solid_fraction` itself has no notion of layers, so
// callers gate it explicitly with `is_bottom` rather than letting a tall
// `terrain_height` bleed rock into higher layers just because `r` still
// falls below the mean terrain radius there
float open_fraction(float3 dir, float r, float mip, bool is_bottom)
{
    return is_bottom ? 1.0 - solid_fraction(dir, r, mip) : 1.0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_divergence(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= velocity.resolution) || thread.z >= 6 * velocity.vertical_resolution)
        return;

    uint3 cell = thread;

    uint3 xm = velocity.get_neighbor(cell, int2(-1, 0));
    uint3 xp = velocity.get_neighbor(cell, int2( 1, 0));
    uint3 ym = velocity.get_neighbor(cell, int2( 0,-1));
    uint3 yp = velocity.get_neighbor(cell, int2( 0, 1));

    float3 dir    = cube_uvw_to_dir(float3(float2(cell.xy + 0.5) / velocity.resolution, cell.z));
    float3 dir_xm = cube_uvw_to_dir(float3(float2(xm.xy + 0.5) / velocity.resolution, xm.z));
    float3 dir_xp = cube_uvw_to_dir(float3(float2(xp.xy + 0.5) / velocity.resolution, xp.z));
    float3 dir_ym = cube_uvw_to_dir(float3(float2(ym.xy + 0.5) / velocity.resolution, ym.z));
    float3 dir_yp = cube_uvw_to_dir(float3(float2(yp.xy + 0.5) / velocity.resolution, yp.z));

    let frame = CubeMapTangentFrame(dir);
    float3 vel_center = velocity.load(cell);
    float vel_xm = frame.to_local(velocity.load_world_space(xm, dir)).x;
    float vel_xp = frame.to_local(velocity.load_world_space(xp, dir)).x;
    float vel_ym = frame.to_local(velocity.load_world_space(ym, dir)).y;
    float vel_yp = frame.to_local(velocity.load_world_space(yp, dir)).y;

    // non-uniform central difference: neighbor spacing varies with where
    // `cell` sits on its face, so the two half-differences are weighted by
    // the actual geodesic distance to each neighbor instead of assumed to
    // both be one texel wide
    float dist_xm = sphere_geodesic_distance(dir, dir_xm);
    float dist_xp = sphere_geodesic_distance(dir, dir_xp);
    float dist_ym = sphere_geodesic_distance(dir, dir_ym);
    float dist_yp = sphere_geodesic_distance(dir, dir_yp);

    // radial term: same (face,u,v) footprint at layer+-1, so no parallel
    // transport is needed (the outward normal is identical at every layer);
    // at a closed boundary the missing neighbor is mirrored with a sign
    // flip so the wall-normal velocity is exactly zero at the wall
    float dr = velocity.thickness / velocity.vertical_resolution;
    uint layer = cell.z / 6;
    float r = 1.0 + dr * (layer + 0.5);
    uint3 zm_cell, zp_cell;
    bool zm_ok = vertical_neighbor(cell, -1, velocity.vertical_resolution, zm_cell);
    bool zp_ok = vertical_neighbor(cell,  1, velocity.vertical_resolution, zp_cell);
    float w_center = vel_center.z;
    float w_zm_raw = zm_ok ? velocity.load(zm_cell).z : -w_center;
    float w_zp_raw = zp_ok ? velocity.load(zp_cell).z : -w_center;

    // sub-grid terrain porosity: a face is only as open as whichever side
    // of it has less rock under it (`min` of the two cells' openness), so
    // partially-blocked faces bleed flow through in proportion instead of
    // an all-or-nothing wall; this is what squeezes flow upward as it
    // crosses a ridge that only partially fills the cells it passes
    // through, rather than only reacting once a whole cell is submerged
    float mip = heightmap_mip_level(velocity.resolution);
    bool self_is_bottom = layer == 0;
    bool below_is_bottom = zm_ok && layer == 1;
    float open_self = open_fraction(dir, r, mip, self_is_bottom);
    float open_xm = min(open_self, open_fraction(dir_xm, r, mip, self_is_bottom));
    float open_xp = min(open_self, open_fraction(dir_xp, r, mip, self_is_bottom));
    float open_ym = min(open_self, open_fraction(dir_ym, r, mip, self_is_bottom));
    float open_yp = min(open_self, open_fraction(dir_yp, r, mip, self_is_bottom));
    float open_zm = zm_ok ? min(open_self, open_fraction(dir, r - dr, mip, below_is_bottom)) : 0.0;
    float open_zp = zp_ok ? open_self : 0.0;

    float vel_xm_eff = lerp(-vel_center.x, vel_xm, open_xm);
    float vel_xp_eff = lerp(-vel_center.x, vel_xp, open_xp);
    float vel_ym_eff = lerp(-vel_center.y, vel_ym, open_ym);
    float vel_yp_eff = lerp(-vel_center.y, vel_yp, open_yp);
    float w_zm = lerp(-w_center, w_zm_raw, open_zm);
    float w_zp = lerp(-w_center, w_zp_raw, open_zp);

    // anelastic divergence constraint div(rho0 * u) = 0: rho0 only varies
    // radially, so it factors out of the horizontal terms as a plain scalar,
    // but the vertical term needs to be in flux form (rho0 sampled at each
    // neighbor layer) to capture d(rho0)/dr; at a closed boundary there's no
    // neighbor layer to evaluate rho0 at, so the ghost reuses this cell's own
    // rho0 (consistent with the w sign-flip ghost above)
    float rho0 = velocity.sample_density(r);
    float rho0_zm = zm_ok ? velocity.sample_density(r - dr) : rho0;
    float rho0_zp = zp_ok ? velocity.sample_density(r + dr) : rho0;

    // scaling the whole cell by `open_self` keeps a mostly-solid cell inert
    // in the solve instead of fighting it toward a spurious target
    divergence.store(cell, open_self * (rho0 * (
        (vel_xp_eff - vel_xm_eff) / (dist_xm + dist_xp) +
        (vel_yp_eff - vel_ym_eff) / (dist_ym + dist_yp)) +
        (rho0_zp * w_zp - rho0_zm * w_zm) / (2.0 * dr)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void step(uint3 thread: SV_DispatchThreadID)
{
    uint res = velocity.resolution >> dst_mip_level;
    if (any(thread.xy >= res) || thread.z >= 6 * velocity.vertical_resolution)
        return;

    uint3 cell = thread;

    uint3 xm = divergence.get_neighbor(cell, int2(-1, 0), dst_mip_level);
    uint3 xp = divergence.get_neighbor(cell, int2( 1, 0), dst_mip_level);
    uint3 ym = divergence.get_neighbor(cell, int2( 0,-1), dst_mip_level);
    uint3 yp = divergence.get_neighbor(cell, int2( 0, 1), dst_mip_level);

    // non-uniform Jacobi stencil for the Laplacian: on a uniform grid every
    // neighbor's weight collapses to 1 (matching the old p*0.25 average),
    // but here each axis's pair of neighbor distances can differ, so the
    // weights are derived from the standard unequal-spacing second-
    // derivative finite-difference formula instead of a plain average
    float3 dir = cube_uvw_to_dir(float3(float2(cell.xy + 0.5) / velocity.resolution, cell.z));
    uint mip_res = velocity.resolution >> dst_mip_level;
    float3 dir_xm = cube_uvw_to_dir(float3(float2(xm.xy + 0.5) / mip_res, xm.z));
    float3 dir_xp = cube_uvw_to_dir(float3(float2(xp.xy + 0.5) / mip_res, xp.z));
    float3 dir_ym = cube_uvw_to_dir(float3(float2(ym.xy + 0.5) / mip_res, ym.z));
    float3 dir_yp = cube_uvw_to_dir(float3(float2(yp.xy + 0.5) / mip_res, yp.z));
    float hxm = sphere_geodesic_distance(dir, dir_xm);
    float hxp = sphere_geodesic_distance(dir, dir_xp);
    float hym = sphere_geodesic_distance(dir, dir_ym);
    float hyp = sphere_geodesic_distance(dir, dir_yp);

    float wxm = 2.0 / (hxm * (hxm + hxp));
    float wxp = 2.0 / (hxp * (hxm + hxp));
    float wym = 2.0 / (hym * (hym + hyp));
    float wyp = 2.0 / (hyp * (hym + hyp));

    // vertical (radial) neighbors: uniform spacing, independent of the
    // horizontal mip level. At a closed boundary there's no flux across the
    // wall, so a Neumann (zero-gradient) condition applies: the ghost value
    // is just a copy of this cell's own (current-iterate) pressure.
    float dr = velocity.thickness / velocity.vertical_resolution;
    uint layer = cell.z / 6;
    float r = 1.0 + dr * (layer + 0.5);
    uint3 zm_cell, zp_cell;
    bool zm_ok = vertical_neighbor(cell, -1, velocity.vertical_resolution, zm_cell);
    bool zp_ok = vertical_neighbor(cell,  1, velocity.vertical_resolution, zp_cell);
    float p_center = pressure_correction[0].load(cell, dst_mip_level);
    float p_zm = pressure_correction[0].load(zm_ok ? zm_cell : cell, dst_mip_level);
    float p_zp = pressure_correction[0].load(zp_ok ? zp_cell : cell, dst_mip_level);
    float wz = 1.0 / (dr * dr);

    // sub-grid terrain porosity (see compute_divergence): scale each
    // neighbor's Laplacian weight by how open that face is, so a
    // mostly-blocked face is effectively dropped from the Jacobi average
    // rather than needing an explicit ghost value -- zeroing its weight
    // already reproduces the zero-flux wall condition
    float mip = heightmap_mip_level(mip_res);
    bool self_is_bottom = layer == 0;
    bool below_is_bottom = zm_ok && layer == 1;
    float open_self = open_fraction(dir, r, mip, self_is_bottom);
    wxm *= min(open_self, open_fraction(dir_xm, r, mip, self_is_bottom));
    wxp *= min(open_self, open_fraction(dir_xp, r, mip, self_is_bottom));
    wym *= min(open_self, open_fraction(dir_ym, r, mip, self_is_bottom));
    wyp *= min(open_self, open_fraction(dir_yp, r, mip, self_is_bottom));
    float wz_zm = zm_ok ? wz * min(open_self, open_fraction(dir, r - dr, mip, below_is_bottom)) : 0.0;
    float wz_zp = zp_ok ? wz * open_self : 0.0;

    float p = 0;
    p += wxm * pressure_correction[0].load(xm, dst_mip_level);
    p += wxp * pressure_correction[0].load(xp, dst_mip_level);
    p += wym * pressure_correction[0].load(ym, dst_mip_level);
    p += wyp * pressure_correction[0].load(yp, dst_mip_level);
    p += wz_zm * p_zm;
    p += wz_zp * p_zp;

    p -= divergence.load(cell, dst_mip_level);

    // a cell fully walled in by terrain has every weight collapse to 0 --
    // fall back to holding its current value rather than dividing by 0
    float total_w = wxm + wxp + wym + wyp + wz_zm + wz_zp;
    pressure_correction[1].store(cell, total_w > 1e-8 ? p / total_w : p_center, dst_mip_level);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resample(uint3 thread: SV_DispatchThreadID)
{
    uint dst_res = velocity.resolution >> dst_mip_level;
    if (any(thread.xy >= dst_res) || thread.z >= 6 * velocity.vertical_resolution)
        return;

    uint3 cell = thread;
    float3 dir = cube_uvw_to_dir(float3((float2(cell.xy) + 0.5) / dst_res, cell.z));

    // pass the cell's own layer through -- omitting it would silently
    // default to layer 0, scrambling every other layer's multires solve
    pressure_correction[1].store(cell, pressure_correction[0].sample_layer(dir, src_mip_level, cell.z / 6), dst_mip_level);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void apply(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= velocity.resolution) || thread.z >= 6 * velocity.vertical_resolution)
        return;

    uint3 cell = thread;

    uint3 xm = pressure_correction[0].get_neighbor(cell, int2(-1, 0));
    uint3 xp = pressure_correction[0].get_neighbor(cell, int2( 1, 0));
    uint3 ym = pressure_correction[0].get_neighbor(cell, int2( 0,-1));
    uint3 yp = pressure_correction[0].get_neighbor(cell, int2( 0, 1));

    float px0 = pressure_correction[0].load(xm);
    float px1 = pressure_correction[0].load(xp);
    float py0 = pressure_correction[0].load(ym);
    float py1 = pressure_correction[0].load(yp);

    // same non-uniform central difference as compute_divergence, so the
    // gradient removed here matches the spacing the divergence/Laplacian
    // were computed with
    float3 dir    = cube_uvw_to_dir(float3(float2(cell.xy + 0.5) / velocity.resolution, cell.z));
    float3 dir_xm = cube_uvw_to_dir(float3(float2(xm.xy + 0.5) / velocity.resolution, xm.z));
    float3 dir_xp = cube_uvw_to_dir(float3(float2(xp.xy + 0.5) / velocity.resolution, xp.z));
    float3 dir_ym = cube_uvw_to_dir(float3(float2(ym.xy + 0.5) / velocity.resolution, ym.z));
    float3 dir_yp = cube_uvw_to_dir(float3(float2(yp.xy + 0.5) / velocity.resolution, yp.z));
    float dist_xm = sphere_geodesic_distance(dir, dir_xm);
    float dist_xp = sphere_geodesic_distance(dir, dir_xp);
    float dist_ym = sphere_geodesic_distance(dir, dir_ym);
    float dist_yp = sphere_geodesic_distance(dir, dir_yp);

    // vertical pressure gradient, same Neumann (copy-of-center) ghost as
    // the Jacobi step above
    float dr = velocity.thickness / velocity.vertical_resolution;
    uint layer = cell.z / 6;
    float r = 1.0 + dr * (layer + 0.5);
    uint3 zm_cell, zp_cell;
    bool zm_ok = vertical_neighbor(cell, -1, velocity.vertical_resolution, zm_cell);
    bool zp_ok = vertical_neighbor(cell,  1, velocity.vertical_resolution, zp_cell);
    float p_center = pressure_correction[0].load(cell);
    float p_zm = zm_ok ? pressure_correction[0].load(zm_cell) : p_center;
    float p_zp = zp_ok ? pressure_correction[0].load(zp_cell) : p_center;

    // sub-grid terrain porosity (see compute_divergence): a blocked face
    // falls back to the Neumann (copy-of-center) ghost instead of the real
    // neighbor pressure -- the same treatment the domain's outer walls
    // already get -- so the pressure force doesn't accelerate fluid through
    // a face that's (partially) rock
    float mip = heightmap_mip_level(velocity.resolution);
    bool self_is_bottom = layer == 0;
    bool below_is_bottom = zm_ok && layer == 1;
    float open_self = open_fraction(dir, r, mip, self_is_bottom);
    float open_xm = min(open_self, open_fraction(dir_xm, r, mip, self_is_bottom));
    float open_xp = min(open_self, open_fraction(dir_xp, r, mip, self_is_bottom));
    float open_ym = min(open_self, open_fraction(dir_ym, r, mip, self_is_bottom));
    float open_yp = min(open_self, open_fraction(dir_yp, r, mip, self_is_bottom));
    float open_zm = zm_ok ? min(open_self, open_fraction(dir, r - dr, mip, below_is_bottom)) : 0.0;
    float open_zp = zp_ok ? open_self : 0.0;

    float px0_eff = lerp(p_center, px0, open_xm);
    float px1_eff = lerp(p_center, px1, open_xp);
    float py0_eff = lerp(p_center, py0, open_ym);
    float py1_eff = lerp(p_center, py1, open_yp);
    float p_zm_eff = lerp(p_center, p_zm, open_zm);
    float p_zp_eff = lerp(p_center, p_zp, open_zp);

    float2 grad = float2((px1_eff - px0_eff) / (dist_xm + dist_xp), (py1_eff - py0_eff) / (dist_ym + dist_yp));
    float grad_z = (p_zp_eff - p_zm_eff) / (2.0 * dr);

    // the correction was solved for as u -= (dt/rho0) * grad(p) folded into
    // a single constant-coefficient Poisson solve (see compute_divergence),
    // so rho0 has to be divided back out here rather than in `step`
    float rho0 = velocity.sample_density(r);

    float3 vel = velocity.load(cell);
    vel.xy -= grad / rho0;
    vel.z -= grad_z / rho0;

    // damp velocity toward zero in proportion to how much of this cell's
    // footprint is rock, so residual momentum doesn't linger inside solid
    // ground (and doesn't reappear as a phantom gust if the terrain there
    // later recedes)
    vel *= open_self;

    velocity.store(cell, vel);
}