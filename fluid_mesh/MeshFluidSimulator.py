import slangpy as spy
import pyobjloader
import numpy as np
import os
from collections import defaultdict

class MeshFluidSimulator:
    def __init__(self, device:spy.Device, obj_file:str):
        self.device = device

        def get_path(p):
            return os.path.join(os.path.dirname(__file__), p)
        self.emit_kernel   = device.create_compute_kernel(device.load_program(get_path("fluid-init.cs.slang"), ["emit_plume"]))
        self.advect_kernel = device.create_compute_kernel(device.load_program(get_path("fluid-advection.cs.slang"), ["advect"]))
        self.compute_vorticity_kernel      = device.create_compute_kernel(device.load_program(get_path("fluid-advection.cs.slang"), ["streamfunction_to_vorticity"]))
        self.compute_streamfunction_kernel = device.create_compute_kernel(device.load_program(get_path("fluid-advection.cs.slang"), ["vorticity_to_streamfunction"]))

        model = pyobjloader.load_model(obj_file)

        assert(len(model.vertex_points.flatten()) <= 0xFFFF)
        assert((model.point_indices <= 0xFFFF).all())

        vertices = np.array(model.vertex_points, dtype=np.float32)
        indices  = np.array(model.point_indices, dtype=np.uint32)

        self.num_vertices  = vertices.shape[0]
        self.num_triangles = indices.shape[0]

        print(f"{self.num_vertices} vertices {self.num_triangles} triangles")

        def edge_key(i0,i1):
            return (np.uint64(min(i0,i1)) << 32) | np.uint64(max(i0,i1))

        adjacencies_list = []
        for _ in range(self.num_vertices):
            adjacencies_list.append(set())
        area_weights = np.zeros(self.num_vertices, dtype=np.float32)
        edge_weights_dict = defaultdict(float)
        for tri in indices:
            assert(tri.shape[0] == 3)
            i0 = tri[0]
            i1 = tri[1]
            i2 = tri[2]

            # adjacencies

            adjacencies_list[i0].add(i1)
            adjacencies_list[i1].add(i0)

            adjacencies_list[i0].add(i2)
            adjacencies_list[i2].add(i0)
            
            adjacencies_list[i2].add(i1)
            adjacencies_list[i1].add(i2)

            # laplacian weights

            #      0
            # e2 /   \ e1
            #   1 --- 2
            #     e0

            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]

            e0 = v2 - v1
            e1 = v0 - v2
            e2 = v1 - v0

            e0 /= np.linalg.norm(e0)
            e1 /= np.linalg.norm(e1)
            e2 /= np.linalg.norm(e2)
            c0 = np.dot(-e1,e2) / np.linalg.norm( np.linalg.cross(e1,e2) )
            c1 = np.dot(-e2,e0) / np.linalg.norm( np.linalg.cross(e2,e0) )
            c2 = np.dot(-e0,e1) / np.linalg.norm( np.linalg.cross(e0,e1) )

            edge_weights_dict[edge_key(i1,i2)] += c0
            edge_weights_dict[edge_key(i0,i2)] += c1
            edge_weights_dict[edge_key(i0,i1)] += c2

            # area weights

            tri_area = np.linalg.norm( np.linalg.cross(e0, e1) ) / 2.0
            area_weights[i0] += tri_area/3.0
            area_weights[i1] += tri_area/3.0
            area_weights[i2] += tri_area/3.0

        assert((np.array([w for w in edge_weights_dict.values()]) > 0).all())

        max_deg = np.max([len(a) for a in adjacencies_list])
        adjacencies  = np.full(shape=(len(vertices), max_deg), fill_value=0xFFFFFFFF, dtype=np.uint32)
        edge_weights = np.zeros(shape=(len(vertices), max_deg), dtype=np.float32)
        for v in range(len(vertices)):
            for i,n in enumerate(adjacencies_list[v]):
                adjacencies[v,i]  = n
                edge_weights[v,i] = edge_weights_dict[edge_key(v,n)]

        self.mesh_vars = {
            "vertices":        device.create_buffer(usage=spy.BufferUsage.shader_resource, data=vertices),
            "indices":         device.create_buffer(usage=spy.BufferUsage.shader_resource, data=indices),
            "adjacencies":     device.create_buffer(usage=spy.BufferUsage.shader_resource, data=adjacencies),
            "vertex_weights":  device.create_buffer(usage=spy.BufferUsage.shader_resource, data=area_weights),
            "edge_weights":    device.create_buffer(usage=spy.BufferUsage.shader_resource, data=edge_weights),
            "psi":             device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "psi_rw":          device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "vorticity":       device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "vorticity_rw":    device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "smoke":           device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "smoke_rw":        device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "num_vertices":    self.num_vertices,
            "num_triangles":   self.num_triangles,
            "num_adjacencies": adjacencies.shape[1],
        }
        self.reset = True

    def setup_ui(self, window:spy.ui.Widget):
        def reset_cb():
            self.reset = True
        self.reset_button = spy.ui.Button(window, "Reset", callback=reset_cb)
        self.jacobi_iterations = spy.ui.DragInt(window, "Jacobi iterations", value=20)
        self.dt = spy.ui.DragFloat(window, "Timestep", 0.01)
        self.emit_plume = spy.ui.CheckBox(window, "Emit plume")

    def step(self, command_encoder:spy.CommandEncoder, dt):
        def swap(n):
            self.mesh_vars[n], self.mesh_vars[f"{n}_rw"] = self.mesh_vars[f"{n}_rw"], self.mesh_vars[n]

        if self.reset:
            for n in ["psi", "vorticity", "smoke"]:
                command_encoder.clear_buffer(self.mesh_vars[n])
                command_encoder.clear_buffer(self.mesh_vars[f"{n}_rw"])
            self.reset = False

        # compute vorticity from streamfunction
        self.compute_vorticity_kernel.dispatch(
            [4096, (self.num_vertices + 4095) // 4096, 1],
            vars={
                "mesh": self.mesh_vars,
            },
            command_encoder=command_encoder
        )
        swap("vorticity")

        # advect vorticity and smoke
        self.advect_kernel.dispatch(
            [4096, (self.num_vertices + 4095) // 4096, 1],
            vars={
                "mesh": self.mesh_vars,
                "dt":   self.dt.value,
            },
            command_encoder=command_encoder
        )
        swap("vorticity")

        # compute streamfunction from advected vorticity
        command_encoder.clear_buffer(self.mesh_vars["psi_rw"])
        for _ in range(self.jacobi_iterations.value):
            swap("psi")
            self.compute_streamfunction_kernel.dispatch(
                [4096, (self.num_vertices + 4095) // 4096, 1],
                vars={
                    "mesh": self.mesh_vars,
                },
                command_encoder=command_encoder
            )

        if self.emit_plume.value:
            self.emit_kernel.dispatch(
                [4096, (self.num_vertices + 4095) // 4096, 1],
                vars={
                    "mesh":         self.mesh_vars,
                    "target_pos":   spy.float3(0,0,1),
                    "target_angle": np.radians(10),
                    "target_dir":   spy.float3(0,1,0),
                },
                command_encoder=command_encoder
            )

        swap("psi")
        swap("smoke")