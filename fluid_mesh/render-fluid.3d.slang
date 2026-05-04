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
    float3 value   : TEXCOORD1;
};

[shader("vertex")]
v2f vs(uint vertexId: SV_VertexID)
{
    uint index = mesh.load_indices(vertexId / 3)[vertexId % 3];
    float3 vertex = mesh.load_vertex(index);

    v2f o;
    o.pos = mul(view_projection, float4(vertex, 1));
    o.prim_id = vertexId / 3;
    switch ((RenderMode)render_mode)
    {
        case RenderMode::Smoke:
            o.value = mesh.smoke[index];
            break;
        case RenderMode::Velocity:
            o.value = compute_velocity_at_vertex(mesh, index, vertex);
            break;
        case RenderMode::Streamfunction:
            o.value = mesh.psi[index];
            break;
        case RenderMode::Vorticity:
            o.value = mesh.vorticity[index];
            break;
    }

    return o;
}

[shader("fragment")]
float4 fs(v2f i, float3 bary : SV_BaryCentrics) : SV_Target
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

    return c;
}