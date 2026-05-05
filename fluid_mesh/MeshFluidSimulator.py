import slangpy as spy
import numpy as np
import os
from collections import defaultdict

def edge_key(i0,i1):
    return (np.uint64(min(i0,i1)) << 32) | np.uint64(max(i0,i1))

def trimesh_laplacian(vertices:np.ndarray, indices:np.ndarray):
    # compute laplacian edge weights

    i0s = indices[:, 0]
    i1s = indices[:, 1]
    i2s = indices[:, 2]
    
    v0s = vertices[i0s]
    v1s = vertices[i1s]
    v2s = vertices[i2s]
    
    e0s = v2s - v1s
    e1s = v0s - v2s
    e2s = v1s - v0s
    e0s /= np.linalg.norm(e0s, axis=1, keepdims=True)
    e1s /= np.linalg.norm(e1s, axis=1, keepdims=True)
    e2s /= np.linalg.norm(e2s, axis=1, keepdims=True)
    
    # compute cotangent weights

    c0s = np.sum(-e1s * e2s, axis=1) / np.linalg.norm(np.cross(e1s, e2s), axis=1)
    c1s = np.sum(-e2s * e0s, axis=1) / np.linalg.norm(np.cross(e2s, e0s), axis=1)
    c2s = np.sum(-e0s * e1s, axis=1) / np.linalg.norm(np.cross(e0s, e1s), axis=1)
    
    edge_weights_dict = defaultdict(float)
    for i, j, c in zip(i1s, i2s, c0s):
        edge_weights_dict[edge_key(i, j)] += c
    for i, j, c in zip(i0s, i2s, c1s):
        edge_weights_dict[edge_key(i, j)] += c
    for i, j, c in zip(i0s, i1s, c2s):
        edge_weights_dict[edge_key(i, j)] += c
    return edge_weights_dict

class MeshFluidSimulator:
    def __init__(self, device:spy.Device, vertices:np.ndarray, indices:np.ndarray):
        self.device = device

        def create_kernel(shader_file, entry):
            path = os.path.join(os.path.dirname(__file__), shader_file)
            return device.create_compute_kernel(device.load_program(path, [entry]))
        self.emit_kernel = create_kernel("fluid-init.cs.slang", "emit_plume")
        self.advect_kernel = create_kernel("fluid-advection.cs.slang", "advect")
        self.compute_vorticity_kernel = create_kernel("fluid-advection.cs.slang", "streamfunction_to_vorticity")
        self.compute_streamfunction_jacobi_kernel = create_kernel("fluid-advection.cs.slang", "vorticity_to_streamfunction_jacobi")
        self.compute_streamfunction_gauss_seidel_kernel = create_kernel("fluid-advection.cs.slang", "vorticity_to_streamfunction_gauss_seidel")

        self.num_vertices = vertices.shape[0]
        self.num_faces    = indices.shape[0]

        # Preprocess mesh
        
        print("Computing laplacian")

        edge_weights_dict = trimesh_laplacian(vertices, indices)
        
        print("Computing adjacencies")

        # compute adjacencies

        adjacencies_list = []
        for _ in range(self.num_vertices):
            adjacencies_list.append(set())
        for face in indices:
            for i in range(len(face)):
                i0 = face[i]
                i1 = face[i-1]
                adjacencies_list[i0].add(i1)
                adjacencies_list[i1].add(i0)

        max_deg = np.max([len(a) for a in adjacencies_list])
        print(f"verts: {self.num_vertices}, prims: {self.num_faces}, max deg: {max_deg}")

        adjacencies  = np.full(shape=(len(vertices), max_deg), fill_value=0xFFFFFFFF, dtype=np.uint32)
        edge_weights = np.zeros(shape=(len(vertices), max_deg), dtype=np.float32)
        for v in range(len(vertices)):
            for i,n in enumerate(adjacencies_list[v]):
                adjacencies[v,i]  = n
                edge_weights[v,i] = edge_weights_dict[edge_key(v,n)]

        # compute area weights
        
        area_weights = np.zeros(self.num_vertices, dtype=np.float32)
        tri_areas = np.linalg.norm(np.cross(vertices[indices[:, 1]] - vertices[indices[:, 0]], vertices[indices[:, 2]] - vertices[indices[:, 0]]), axis=1) / 2.0
        np.add.at(area_weights, indices[:, 0], tri_areas / 3.0)
        np.add.at(area_weights, indices[:, 1], tri_areas / 3.0)
        np.add.at(area_weights, indices[:, 2], tri_areas / 3.0)

        assert((np.array([w for w in edge_weights_dict.values()]) > 0).all())

        print("Computing red-black coloring")

        # compute red-black coloring for gauss-siedel
        colored_vertices = np.empty(self.num_vertices, dtype=np.uint32)
        colored_vertices[0] = 0
        front_idx = 1
        back_idx = 0
        color = -np.ones(self.num_vertices, dtype=np.int8)
        color[0] = 0
        queue = [0]
        while queue:
            i = queue.pop()
            for j in adjacencies_list[i]:
                if color[j] == -1:
                    c = 1 - color[i]
                    color[j] = c
                    queue.append(j)
                    if c == 0:
                        colored_vertices[front_idx] = j
                        front_idx += 1
                    else:
                        colored_vertices[-back_idx - 1] = j
                        back_idx += 1
        assert(front_idx + back_idx == len(vertices))
        self.color_range = front_idx
        
        print("Done preprocessing")

        self.mesh_vars = {
            "vertices":         device.create_buffer(usage=spy.BufferUsage.shader_resource, data=vertices),
            "indices":          device.create_buffer(usage=spy.BufferUsage.shader_resource, data=indices),
            "adjacencies":      device.create_buffer(usage=spy.BufferUsage.shader_resource, data=adjacencies),
            "colored_vertices": device.create_buffer(usage=spy.BufferUsage.shader_resource, data=colored_vertices),
            "vertex_weights":   device.create_buffer(usage=spy.BufferUsage.shader_resource, data=area_weights),
            "edge_weights":     device.create_buffer(usage=spy.BufferUsage.shader_resource, data=edge_weights),
            "psi":              device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "psi_rw":           device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "vorticity":        device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "vorticity_rw":     device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "smoke":            device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "smoke_rw":         device.create_buffer(element_count=len(vertices), struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "num_vertices":     self.num_vertices,
            "num_faces":        self.num_faces,
            "num_adjacencies":  adjacencies.shape[1],
        }
        self.reset = True

    def setup_ui(self, window:spy.ui.Widget):
        self.step_once = False
        def reset_cb():
            self.reset = True
        def step_cb():
            self.step_once = True
        self.paused = spy.ui.CheckBox(window, "Pause")
        spy.ui.Button(window, "Step", callback=step_cb)
        self.reset_button = spy.ui.Button(window, "Reset", callback=reset_cb)
        self.solver = spy.ui.ComboBox(window, "Solver", 0, items=["Jacobi", "Gauss-Seidel"])
        self.jacobi_iterations = spy.ui.DragInt(window, "Jacobi iterations", value=10)
        self.overrelaxation = spy.ui.SliderFloat(window, "Overrelaxation", value=1.25, min=1, max=2)
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

        if self.paused.value and not self.step_once:
            return
        self.step_once = False

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
        if self.solver.value == 0:
            # Jacobi
            for _ in range(self.jacobi_iterations.value):
                swap("psi")
                self.compute_streamfunction_jacobi_kernel.dispatch(
                    [4096, (self.num_vertices + 4095) // 4096, 1],
                    vars={
                        "mesh": self.mesh_vars,
                    },
                    command_encoder=command_encoder
                )
        else:
            # Gauss-Seidel
            for _ in range(self.jacobi_iterations.value):
                for r in [ [0, self.color_range], [self.color_range, self.num_vertices] ]:
                    self.compute_streamfunction_gauss_seidel_kernel.dispatch(
                        [4096, ((r[1] - r[0]) + 4095) // 4096, 1],
                        vars={
                            "mesh": self.mesh_vars,
                            "omega_sor": self.overrelaxation.value,
                            "color_range": r
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