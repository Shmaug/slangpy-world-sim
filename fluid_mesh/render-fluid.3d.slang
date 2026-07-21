import Math;
import FluidMesh;

uniform FluidMesh mesh;

uniform float4x4 view_projection;
uniform uint     render_mode;
uniform uint     level;
uniform float3 mouse_ray_org;
uniform float3 mouse_ray_dir;

enum RenderMode
{
    Smoke = 0,
    Velocity = 1,
    Mesh = 2
}

struct v2f
{
    float4 pos : SV_Position;
    uint   face_index : TEXCOORD0;
    float3 pos_w : TEXCOORD1;
};

[shader("vertex")]
v2f vs(uint vertexId: SV_VertexID)
{
    uint face_index = vertexId / 3;
    face_index += mesh.level_face_offsets[level];

    uint vertex_index = mesh.load_face(face_index)[vertexId % 3];
    float3 vertex = mesh.load_vertex(vertex_index);

    v2f o;
    o.pos = mul(view_projection, float4(vertex, 1));
    o.face_index = face_index;
    o.pos_w = vertex;

    return o;
}

[shader("fragment")]
float4 fs(v2f i, float3 i_bary : SV_BaryCentrics) : SV_Target
{
    float4 c = float4(0, 0, 0, 1);

    uint3 face = mesh.load_face(i.face_index);
    float3x3 vertices = mesh.load_face_vertices(face);

    switch ((RenderMode)render_mode)
    {
        case RenderMode::Smoke:
        {
            float smoke = mesh.sample_smoke(normalize(i.pos_w), i.face_index, vertices);
            c.rgb = saturate(float3(1 - exp(smoke.x), 1 - exp(-smoke.x), 0.1));
            break;
        }
        case RenderMode::Velocity:
        {
            float3 bary = triangle_barycentrics(normalize(i.pos_w), vertices);
            float3 vel = mesh.sample_velocity(bary, face, vertices, mesh.load_face_edges(i.face_index));
            c.rgb = 5 * vel + 0.5;
            break;
        }
        case RenderMode::Mesh:
        {
            c.rgb = float3(uint3(pcg(i.face_index), pcg(pcg(i.face_index)), pcg(pcg(pcg(i.face_index)))) % 255) / 255;
            c.rgb = lerp(c.rgb, 0.5, 0.75);
            
            float2 t;
            if (intersect_sphere(mouse_ray_org, mouse_ray_dir, 1, t))
            {
                float3 selected_pos = mouse_ray_org + mouse_ray_dir * t.x;

                uint selected_face_index;
                uint3 selected_face;
                float3x3 selected_face_vertices;
                if (mesh.find_face(selected_pos, selected_face_index, selected_face, selected_face_vertices, level))
                {
                    if (i.face_index == selected_face_index)
                    {
                        c.rgb = 1;

                        [ForceUnroll]
                        for (uint v = 0; v < 3; v++)
                        {
                            float s = acos(clamp(dot(vertices[v], normalize(i.pos_w)), -1, 1)) / radians(0.1);

                            float3 vc = 0;
                            vc[v] = 1;

                            c.rgb = lerp(c.rgb, vc, saturate(10 * exp(-s)));
                        }
                    }
                    else
                    {
                        uint3 adj = mesh.load_adjacent_faces(selected_face_index);

                        [ForceUnroll]
                        for (uint a = 0; a < 3; a++)
                        {
                            if (i.face_index == adj[a])
                            {
                                c = 0;
                                c[a] = 1;
                            }
                        }
                    }
                }
            }

            break;
        }
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