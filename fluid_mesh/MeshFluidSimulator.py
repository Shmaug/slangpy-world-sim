import slangpy as spy
import numpy as np
import os

class MeshFluidSimulator:
    def __init__(self, device:spy.Device, widget:spy.ui.Widget|None = None):
        self.device = device

        def create_kernel(shader_file, entry):
            path = os.path.join(os.path.dirname(__file__), shader_file)
            program = self.device.load_program(path, [entry])
            return self.device.create_compute_kernel(program)
        
        self.emit_kernel = create_kernel("fluid-init.cs.slang", "emit_plume")
        self.advect_kernel = create_kernel("fluid-advection.cs.slang", "advect")
        self.swap_velocity_kernel = create_kernel("fluid-advection.cs.slang", "swap_velocity")

        self.subdivision_levels = 7
        self.create_mesh()

        self.reset = True
        self.step_once = False

        if widget is not None:
            def reset_cb():
                self.reset = True
            def step_cb():
                self.step_once = True
            def level_cb(value):
                self.subdivision_levels = min(max(1,self.subdivision_level_ui.value), 10)
                self.create_mesh()
            self.paused = spy.ui.CheckBox(widget, "Pause")
            spy.ui.Button(widget, "Step", callback=step_cb)
            self.reset_button = spy.ui.Button(widget, "Reset", callback=reset_cb)
            self.subdivision_level_ui = spy.ui.DragInt(widget, "Subdivision level", value=7, min=1, max=10, callback=level_cb)
            self.solver_iterations = spy.ui.DragInt(widget, "Solver iterations", value=100)
            self.dt = spy.ui.DragFloat(widget, "Timestep", 0.01)
            self.emit_plume = spy.ui.CheckBox(widget, "Emit plume")

    def create_mesh(self):
        print("Subdividing mesh")

        def edge_key(i0,i1):
            return (np.uint64(min(i0,i1)) << 32) | np.uint64(max(i0,i1))

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
        level_face_offsets = [0]
        self.level_face_counts = [len(faces)]

        for level in range(1, self.subdivision_levels):
            edge_midpoints = {} # new vertices added per-edge
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
                m0 = edge_midpoints[edge_key(i0,i1)]
                m1 = edge_midpoints[edge_key(i1,i2)]
                m2 = edge_midpoints[edge_key(i2,i0)]

                face_children[face_index] = len(faces)
                for f in [
                    [ i0, m0, m2 ],
                    [ m0, i1, m1 ],
                    [ m0, m1, m2 ],
                    [ m2, m1, i2 ]
                ]:
                    faces.append(f)
                    face_children.append(0xFFFFFFFF)
            
            self.level_face_counts.append(20*(1<<(2*level)))

        vertices = np.array(vertices, np.float32)
        faces = np.array(faces, np.uint32)
        face_children = np.array(face_children, np.uint32)
        level_face_offsets = np.array(level_face_offsets, np.uint32)
        self.level_face_counts = np.array(self.level_face_counts, np.uint32)

        # extract edge and adjacency info

        edge_map = {}
        edges = []
        face_edges = np.full((faces.shape[0], 3), 0xFFFFFFFF, np.uint32)
        face_adjacencies = np.full((faces.shape[0], 3), 0xFFFFFFFF, np.uint32)
        for face_index in range(faces.shape[0]):
            face = faces[face_index]
            for j in range(3):
                v0, v1 = face[(j+1)%3], face[(j+2)%3]
                e = edge_key(v0,v1)
                if e in edge_map:
                    edge_index, neighbor_face_index, neighbor_j = edge_map[e]
                    face_edges[face_index][j] = edge_index
                    face_adjacencies[face_index][j] = neighbor_face_index
                    face_adjacencies[neighbor_face_index][neighbor_j] = face_index
                else:
                    edge_index = len(edges)
                    edges.append([v0,v1])
                    edge_map[e] = (edge_index, face_index, j)
                    face_edges[face_index][j] = edge_index

        edges = np.array(edges, np.uint32)

        self.edge_count = edges.shape[0]

        print("Done preprocessing")

        self.mesh_vars = {       
            "vertices":       self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=vertices),
            "faces":          self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=faces),
            "edges":          self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=edges),
            "face_children":  self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=face_children),
            "face_edges":     self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=face_edges),
            "face_adjacencies": self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=face_adjacencies),
            "level_face_offsets": self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=level_face_offsets),
            "level_face_counts": self.device.create_buffer(usage=spy.BufferUsage.shader_resource, data=self.level_face_counts),
            "edge_count": self.edge_count,
            "levels": self.subdivision_levels,
            
            "velocity":         self.device.create_buffer(element_count=2*edges.shape[0], struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "velocity_rw":      self.device.create_buffer(element_count=2*edges.shape[0], struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "smoke":            self.device.create_buffer(element_count=faces.shape[0], struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
            "smoke_rw":         self.device.create_buffer(element_count=faces.shape[0], struct_size=4, usage=spy.BufferUsage.shader_resource|spy.BufferUsage.unordered_access),
        }

        self.reset = True

    def step(self, command_encoder:spy.CommandEncoder, dt):
        def swap(n):
            self.mesh_vars[n], self.mesh_vars[f"{n}_rw"] = self.mesh_vars[f"{n}_rw"], self.mesh_vars[n]

        if self.reset:
            for n in ["velocity", "smoke"]:
                command_encoder.clear_buffer(self.mesh_vars[n])
                command_encoder.clear_buffer(self.mesh_vars[f"{n}_rw"])
            self.reset = False

        if self.paused.value and not self.step_once:
            return
        self.step_once = False

        # advect velocity and smoke
        # self.advect_kernel.dispatch(
        #     [4096, (self.level_face_counts[-1] + 4095) // 4096, 1],
        #     vars={
        #         "mesh": self.mesh_vars,
        #         "dt": self.dt.value,
        #         "level": self.subdivision_levels-1,
        #     },
        #     command_encoder=command_encoder
        # )
        # swap("velocity")
        # swap("smoke")

        command_encoder.copy_buffer(self.mesh_vars["smoke_rw"], 0, self.mesh_vars["smoke"], 0, self.mesh_vars["smoke"].size)

        if self.emit_plume.value:
            self.emit_kernel.dispatch(
                [4096, (self.level_face_counts[-1] + 4095) // 4096, 1],
                vars={
                    "mesh":         self.mesh_vars,
                    "target_pos":   spy.float3(0,0,1),
                    "target_angle": np.radians(10),
                    "target_dir":   spy.float3(0,1,0),
                },
                command_encoder=command_encoder
            )
        
        self.swap_velocity_kernel.dispatch(
            [4096, (self.edge_count + 4095) // 4096, 1],
            vars={
                "mesh": self.mesh_vars,
            },
            command_encoder=command_encoder
        )
        swap("smoke")