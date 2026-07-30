import Math;
import SphericalField;

uniform SphericalScalarField smoke;
uniform SphericalVelocityField velocity;
uniform SphericalScalarField smoke_out;
uniform SphericalVelocityField velocity_out;

uniform float3 start_dir;
uniform float  radius;
uniform float3 end_dir;
uniform float  speed;
uniform float  vertical_speed;

[shader("compute")]
[numthreads(8, 8, 1)]
void emit_drag(uint3 thread: SV_DispatchThreadID)
{
    if (any(thread.xy >= velocity.resolution) || thread.z >= 6 * velocity.vertical_resolution)
        return;

    CubeCoord cell = CubeCoord(thread.xy, thread.z % 6, thread.z / 6, 0);
    float cloud = smoke.load<1>(cell);
    float3 vel = velocity.load(cell);

    // only inject at the ground layer -- the plume rises into higher
    // layers via radial advection instead of being stamped into all of them
    uint layer = thread.z / 6;
    if (layer > 0 && radius > 0)
    {
        float3 pos = cube_uvw_to_dir(float3((thread.xy + 0.5) / velocity.resolution, thread.z%6));

        // axis of the great circle through start_dir and end_dir
        float3 drag_axis_raw = cross(start_dir, end_dir);
        float drag_axis_len = length(drag_axis_raw);
        float3 drag_axis = drag_axis_len > 1e-8 ? drag_axis_raw / drag_axis_len : float3(0, 0, 0);

        // tangent at start_dir pointing towards end_dir along the geodesic
        float3 drag_tangent = drag_axis_len > 1e-8 ? cross(drag_axis, start_dir) : float3(0, 0, 0);
        float path_angle = sphere_geodesic_distance(start_dir, end_dir);

        // point on that geodesic closest to pos: project pos onto the plane
        // orthogonal to the axis, renormalize back onto the sphere, then clamp
        // its position along the arc so it can't slide past start_dir/end_dir
        float3 drag_pos = start_dir;
        if (drag_axis_len > 1e-8)
        {
            float3 closest = safe_normalize(pos - drag_axis * dot(pos, drag_axis));
            float t = atan2(dot(closest, drag_tangent), dot(closest, start_dir));
            float t_clamped = clamp(t, 0.0, path_angle);
            drag_pos = cos(t_clamped) * start_dir + sin(t_clamped) * drag_tangent;
        }

        // Distance from emission point
        float angle = acos(clamp(dot(drag_pos, pos), -1.0, 1.0));

        // Create a plume at target_pos
        if (radius > 0 && angle < radius * 2.0)
        {
            float normalized_dist = angle / (radius * 2.0);
            float gaussian = exp(-normalized_dist * normalized_dist * 4.0);

            // tangent at start_dir pointing towards end_dir along the geodesic,
            // scaled by the path's arc length so its magnitude is a velocity
            float3 target_vel = drag_axis_len > 1e-8 ? path_angle * cross(drag_axis, start_dir) : float3(0, 0, 0);

            float3 kick = CubeMapTangentFrame(pos).to_local3(target_vel);
            kick.xy *= speed;
            kick.z *= vertical_speed;
            
            cloud += gaussian;
            vel += gaussian * kick;
        }
    }

    smoke_out.store(cell, cloud);
    velocity_out.store(cell, vel);
}