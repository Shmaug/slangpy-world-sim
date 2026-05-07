import Math;
import FluidMesh;

uniform FluidMesh mesh;

uniform float4x4 view_projection;
uniform uint     render_mode;

enum RenderMode
{
    Smoke = 0,
    Velocity = 1,
    Streamfunction = 2,
    Vorticity = 3,
}

struct v2f
{
    float4 pos : SV_Position;
    uint   prim_id : TEXCOORD0;
    float3 pos_w   : TEXCOORD1;
    float3 value   : TEXCOORD2;
};

[shader("vertex")]
v2f vs(uint vertexId: SV_VertexID)
{
    uint vertex_index = mesh.load_indices(mesh.levels-1, vertexId / 3)[vertexId % 3];
    float3 vertex = mesh.load_vertex(vertex_index);

    uint offset = mesh.vertex_offsets[mesh.levels-1];

    v2f o;
    o.pos = mul(view_projection, float4(vertex, 1));
    o.prim_id = vertexId / 3;
    o.pos_w = vertex;
    switch ((RenderMode)render_mode)
    {
        case RenderMode::Smoke:
            o.value = mesh.smoke[vertex_index];
            break;
        case RenderMode::Velocity:
            o.value = mesh.compute_velocity_at_vertex(vertex_index, vertex);
            break;
        case RenderMode::Streamfunction:
            o.value = mesh.psi[offset + vertex_index];
            break;
        case RenderMode::Vorticity:
            o.value = mesh.vorticity[offset + vertex_index];
            break;
    }

    return o;
}

[shader("fragment")]
float4 fs(v2f i, float3 i_bary : SV_BaryCentrics) : SV_Target
{
    float4 c = float4(0,0,0,1);

    if (render_mode == RenderMode::Velocity)
    {
        c.rgb = 5*i.value + 0.5;
    }
    else
    {
        c.rgb = saturate(float3(1 - exp(i.value.x), 1 - exp(-i.value.x), 0.1));
    }

    /*
    uint3 face;
    float3 bary;
    if (!mesh.find_face(normalize(i.pos_w), face, bary))
    {
        c.rgb = float3(1,0,0);
    }
    else
    {
        float3x3 verts = mesh.load_face_vertices(face);
        float3 pos = 
            bary.x * verts[0] +
            bary.y * verts[1] +
            bary.z * verts[2];
        c.rgb = 100*length(pos - i.pos_w);
    }
    //*/

    return c;
}