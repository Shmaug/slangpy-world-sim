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
    def __init__(self, device:spy.Device):
        self.device = device

        self.create_mesh(8)

        def create_kernel(shader_file, entry):
            path = os.path.join(os.path.dirname(__file__), shader_file)
            return device.create_compute_kernel(device.load_program(path, [entry]))
        self.emit_kernel = create_kernel("fluid-init.cs.slang", "emit_plume")
        self.advect_kernel = create_kernel("fluid-advection.cs.slang", "advect")
        self.compute_vorticity_kernel = create_kernel("fluid-advection.cs.slang", "streamfunction_to_vorticity")
        self.compute_streamfunction_jacobi_kernel = create_kernel("fluid-advection.cs.slang", "vorticity_to_streamfunction_jacobi")
        self.compute_streamfunction_gauss_seidel_kernel = create_kernel("fluid-advection.cs.slang", "vorticity_to_streamfunction_gauss_seidel")

    def create_mesh(self, levels):
        print("Subdividing mesh")

        vertices = [
            np.array([ 0.000000, -1.000000, 0.000000 ], dtype=np.float32),
            np.array([ 0.723600, -0.447215, 0.525720 ], dtype=np.float32),
            np.array([ -0.276385, -0.447215, 0.850640 ], dtype=np.float32),
            np.array([ -0.894425, -0.447215, 0.000000 ], dtype=np.float32),
            np.array([ -0.276385, -0.447215, -0.850640 ], dtype=np.float32),
            np.array([ 0.723600, -0.447215, -0.525720 ], dtype=np.float32),
            np.array([ 0.276385, 0.447215, 0.850640 ], dtype=np.float32),
            np.array([ -0.723600, 0.447215, 0.525720 ], dtype=np.float32),
            np.array([ -0.723600, 0.447215, -0.525720 ], dtype=np.float32),
            np.array([ 0.276385, 0.447215, -0.850640 ], dtype=np.float32),
            np.array([ 0.894425, 0.447215, 0.000000 ], dtype=np.float32),
            np.array([ 0.000000, 1.000000, 0.000000 ], dtype=np.float32),
        ]
        tree_faces = [ [a-1, b-1, c-1] for a,b,c in [
            [ 1, 2, 3 ],
            [ 2, 1, 6 ],
            [ 1, 3, 4 ],
            [ 1, 4, 5 ],
            [ 1, 5, 6 ],
            [ 2, 6, 11 ],
            [ 3, 2, 7 ],
            [ 4, 3, 8 ],
            [ 5, 4, 9 ],
            [ 6, 5, 10 ],
            [ 2, 11, 7 ],
            [ 3, 7, 8 ],
            [ 4, 8, 9 ],
            [ 5, 9, 10 ],
            [ 6, 10, 11 ],
            [ 7, 11, 12 ],
            [ 8, 7, 12 ],
            [ 9, 8, 12 ],
            [ 10, 9, 12 ],
            [ 11, 10, 12 ],
        ] ]
        tree_children = [ np.full(4, 0xFFFFFFFF) for _ in range(len(tree_faces)) ]
        tree_payloads = list(range(len(tree_faces)))

        indices = [ [a,b,c] for a,b,c in tree_faces ]
        last_nodes = list(range(len(tree_faces)))

        for _ in range(1,levels):
            indices.clear()
            new_nodes = []
            edge_midpoints = {}
            for i in last_nodes:
                face = tree_faces[i]

                # add new vertices
                for j in range(3):
                    j0,j1 = face[j],face[(j+1)%3]
                    e = edge_key(j0,j1)
                    if e not in edge_midpoints:
                        new_vertex = len(vertices)
                        edge_midpoints[e] = new_vertex
                        v = (vertices[j0] + vertices[j1])*0.5
                        v /= np.linalg.norm(v) # project to sphere
                        vertices.append(v)

                #     i0
                #    /  \
                #   m0 - m2
                #  /  \  / \
                # i1 --m1-- i2

                i0,i1,i2 = face
                m0,m1,m2 = edge_midpoints[edge_key(i0,i1)], edge_midpoints[edge_key(i1,i2)], edge_midpoints[edge_key(i2,i0)]

                # add new faces

                for j,new_face in enumerate([
                    [ i0, m0, m2 ],
                    [ m0, i1, m1 ],
                    [ m0, m1, m2 ],
                    [ m2, m1, i2 ]
                ]):
                    new_face_index = len(indices)
                    new_node_index = len(tree_faces)
                    # add face to mesh
                    indices.append(new_face)
                    # add node to tree
                    tree_children[i][j] = new_node_index
                    tree_faces.append(new_face)
                    tree_children.append(np.full(4, 0xFFFFFFFF))
                    tree_payloads.append(new_face_index)
                    # track new nodes for next iteration
                    new_nodes.append(new_node_index)
            last_nodes = new_nodes

        vertices = np.array(vertices, np.float32)
        indices = np.array(indices, np.uint32)

        self.num_vertices = vertices.shape[0]
        self.num_faces = indices.shape[0]

        # Preprocess mesh
        
        print("Computing laplacian")

        edge_weights_dict = trimesh_laplacian(vertices, indices)
        
        print("Computing adjacencies")

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
        assert(max_deg <= 6)
        print(f"verts: {self.num_vertices}, prims: {self.num_faces}, max deg: {max_deg}")

        adjacencies  = np.full(shape=(self.num_vertices, max_deg), fill_value=0xFFFFFFFF, dtype=np.uint32)
        edge_weights = np.zeros(shape=(self.num_vertices, max_deg), dtype=np.float32)
        for v in range(self.num_vertices):
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
        assert(front_idx + back_idx == self.num_vertices)
        self.color_range = front_idx
        
        print("Done preprocessing")

        self.mesh_vars = {
            "node_vertex_indices": self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=np.array(tree_faces, dtype=np.uint32)),
            "node_children":       self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=np.array(tree_children, dtype=np.uint32)),
            "node_payloads":       self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=np.array(tree_payloads, dtype=np.uint32)),
            "vertices":         self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=vertices),
            "indices":          self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=indices),
            "adjacencies":      self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=adjacencies),
            "colored_vertices": self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=colored_vertices),
            "vertex_weights":   self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=area_weights),
            "edge_weights":     self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=edge_weights),
            "psi":              self.device.create_buffer(element_count=self.num_vertices, struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "psi_rw":           self.device.create_buffer(element_count=self.num_vertices, struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "vorticity":        self.device.create_buffer(element_count=self.num_vertices, struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "vorticity_rw":     self.device.create_buffer(element_count=self.num_vertices, struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "smoke":            self.device.create_buffer(element_count=self.num_vertices, struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "smoke_rw":         self.device.create_buffer(element_count=self.num_vertices, struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "num_vertices":     self.num_vertices,
            "num_faces":        self.num_faces,
            "levels": levels,
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