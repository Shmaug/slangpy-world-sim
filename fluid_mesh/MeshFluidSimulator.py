import slangpy as spy
import numpy as np
import os
from collections import defaultdict

def edge_key(i0,i1):
    return (np.uint64(min(i0,i1)) << 32) | np.uint64(max(i0,i1))

def process_mesh(vertices:np.ndarray, indices:np.ndarray):
    print("Computing laplacian edge weights")

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

    print("Computing adjacencies")

    num_vertices = vertices.shape[0]

    adjacencies_list = []
    for _ in range(num_vertices):
        adjacencies_list.append(set())
    for face in indices:
        for i in range(len(face)):
            i0 = face[i]
            i1 = face[i-1]
            adjacencies_list[i0].add(i1)
            adjacencies_list[i1].add(i0)

    max_deg = np.max([len(a) for a in adjacencies_list])
    print(f"verts: {num_vertices}, prims: {indices.shape[0]}, max deg: {max_deg}")
    assert(max_deg <= 6)

    adjacencies  = np.full(shape=(num_vertices, 6), fill_value=0xFFFFFFFF, dtype=np.uint32)
    edge_weights = np.zeros(shape=(num_vertices, 6), dtype=np.float32)
    for v in range(num_vertices):
        for i,n in enumerate(adjacencies_list[v]):
            adjacencies[v,i]  = n
            edge_weights[v,i] = edge_weights_dict[edge_key(v,n)]

    print("Computing vertex area weights")

    area_weights = np.zeros(num_vertices, dtype=np.float32)
    tri_areas = np.linalg.norm(np.cross(v1s - v0s, v2s - v0s), axis=1) / 2.0
    np.add.at(area_weights, i0s, tri_areas / 3.0)
    np.add.at(area_weights, i1s, tri_areas / 3.0)
    np.add.at(area_weights, i2s, tri_areas / 3.0)

    assert((np.array([w for w in edge_weights_dict.values()]) > 0).all())

    return adjacencies, edge_weights, area_weights

class MeshFluidSimulator:
    def __init__(self, device:spy.Device):
        self.device = device

        def create_kernel(shader_file, entry):
            path = os.path.join(os.path.dirname(__file__), shader_file)
            program = self.device.load_program(path, [entry])
            return self.device.create_compute_kernel(program)
        
        self.emit_kernel = create_kernel("fluid-init.cs.slang", "emit_plume")
        self.advect_kernel = create_kernel("fluid-advection.cs.slang", "advect")
        self.compute_vorticity_kernel = create_kernel("fluid-advection.cs.slang", "streamfunction_to_vorticity")
        self.compute_streamfunction_kernel = create_kernel("fluid-advection.cs.slang", "vorticity_to_streamfunction")

        self.subdivision_levels = 7
        self.create_mesh()

    def create_mesh(self):
        print("Subdividing mesh")

        # default icosphere
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
        
        faces = [ [a-1, b-1, c-1] for a,b,c in [
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
        face_children = [ 0xFFFFFFFF ] * len(faces)

        level_faces = [ [ [a,b,c] for a,b,c in faces ] ]
        level_face_offsets = [0]
        level_vertex_offsets = [0]

        adjacencies, edge_weights, area_weights = process_mesh(np.array(vertices, np.float32), np.array(level_faces[-1], np.uint32))

        for _ in range(1, self.subdivision_levels):
            new_faces = []
            edge_midpoints = {}
            level_vertex_offsets.append(adjacencies.shape[0])
            level_face_offsets.append(len(faces))
            for face_index in range(level_face_offsets[-2], level_face_offsets[-1]):
                face = faces[face_index]

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

                # add new faces

                #     i0
                #    /  \
                #   m0 - m2
                #  /  \  / \
                # i1 --m1-- i2

                i0,i1,i2 = face
                m0,m1,m2 = edge_midpoints[edge_key(i0,i1)], edge_midpoints[edge_key(i1,i2)], edge_midpoints[edge_key(i2,i0)]

                face_children[face_index] = len(faces)
                for f in [
                    [ i0, m0, m2 ],
                    [ m0, i1, m1 ],
                    [ m0, m1, m2 ],
                    [ m2, m1, i2 ]
                ]:
                    new_faces.append(f)
                    faces.append(f)
                    face_children.append(0xFFFFFFFF)
            
            level_faces.append(new_faces)

            level_adjacencies, level_edge_weights, level_area_weights = process_mesh(np.array(vertices, np.float32), np.array(new_faces, np.uint32))

            adjacencies = np.vstack((adjacencies, level_adjacencies))
            edge_weights = np.vstack((edge_weights, level_edge_weights))
            area_weights = np.hstack((area_weights, level_area_weights))

        vertices = np.array(vertices, np.float32)
        faces = np.array(faces, np.uint32)
        face_children = np.array(face_children, np.uint32)

        self.num_vertices = vertices.shape[0]
        self.num_faces = len(level_faces[-1])

        print("Done preprocessing")

        self.mesh_vars = {       
            "vertices":       self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=vertices),
            "faces":          self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=faces),
            "face_children":  self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=face_children),
            "face_offsets":   self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=np.array(level_face_offsets, np.uint32)),
            "vertex_offsets": self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=np.array(level_vertex_offsets, np.uint32)),
            "adjacencies":    self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=adjacencies),
            "vertex_weights": self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=area_weights),
            "edge_weights":   self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=edge_weights),
            "psi":            self.device.create_buffer(element_count=adjacencies.shape[0], struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "psi_rw":         self.device.create_buffer(element_count=adjacencies.shape[0], struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "vorticity":      self.device.create_buffer(element_count=adjacencies.shape[0], struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "vorticity_rw":   self.device.create_buffer(element_count=adjacencies.shape[0], struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "smoke":          self.device.create_buffer(element_count=self.num_vertices, struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "smoke_rw":       self.device.create_buffer(element_count=self.num_vertices, struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "num_vertices":   self.num_vertices,
            "levels":         self.subdivision_levels,
        }
        self.reset = True

    def setup_ui(self, window:spy.ui.Widget):
        self.step_once = False
        def reset_cb():
            self.reset = True
        def step_cb():
            self.step_once = True
        def level_cb(value):
            self.subdivision_levels = min(max(1,self.subdivision_level_ui.value), 10)
            self.create_mesh()
        self.paused = spy.ui.CheckBox(window, "Pause")
        spy.ui.Button(window, "Step", callback=step_cb)
        self.reset_button = spy.ui.Button(window, "Reset", callback=reset_cb)
        self.subdivision_level_ui = spy.ui.DragInt(window, "Subdivision level", value=7, min=1, max=10, callback=level_cb)
        self.jacobi_iterations = spy.ui.DragInt(window, "Solver iterations", value=10)
        self.multiresolution = spy.ui.CheckBox(window, "Multiresolution solver")
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
        for _ in range(self.jacobi_iterations.value):
            swap("psi")
            self.compute_streamfunction_kernel.dispatch(
                [4096, (self.num_vertices + 4095) // 4096, 1],
                vars={
                    "mesh": self.mesh_vars,
                    "level": self.subdivision_levels-1
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