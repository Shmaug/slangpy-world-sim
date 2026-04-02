import slangpy as spy
import math

class InputState:
    def __init__(self):
        self.state = {}
        
    def on_keyboard_event(self, event: spy.KeyboardEvent, has_focus):
        if has_focus and event.is_key_press(): self.state[event.key] = True
        if event.is_key_release(): self.state[event.key] = False
        
    def on_mouse_event(self, event: spy.MouseEvent, has_focus):
        if event.is_move(): self.state["mouse"] = event.pos
        if has_focus and event.is_button_down(): self.state[event.button] = True
        if has_focus and event.is_scroll(): self.state["scroll"] = event.scroll.y
        if event.is_button_up(): self.state[event.button] = False

    def update(self):
        self.state["scroll"] = 0

    def get(self, key):
        return self.state[key] if key in self.state else None

    def is_down(self, key):
        return key in self.state and self.state[key]

class Camera:
    def __init__(self):
        self.position = spy.float3(0,0,1.5)
        self.rotation = spy.float2(0,0)
        self.fovY = spy.math.radians(60)
        self.drag_start = None
        self.drag_start_rotation = None
        self.move_speed = 0.1
        self.sensitivity = .002

    def get_rotation(self):
        ry = spy.math.rotate(spy.float4x4(), self.rotation[0], spy.float3(0,1,0))
        rx = spy.math.rotate(spy.float4x4(), self.rotation[1], spy.float3(1,0,0))
        return spy.math.mul(ry, rx)
    
    def camera_to_world(self):
        return spy.math.mul(spy.math.translate(spy.float4x4(), self.position), self.get_rotation())

    def projection(self, aspect, nearZ = 0.01, farZ = 100):
        return spy.math.perspective(self.fovY, aspect, nearZ, farZ)

    def update(self, input_state, dt):
        mouse = input_state.get("mouse")
        if input_state.is_down(spy.MouseButton.left) and mouse is not None:
            if self.drag_start is None:
                self.drag_start = mouse
                self.drag_start_rotation = self.rotation
            delta = (mouse - self.drag_start) * self.sensitivity
            self.rotation = self.drag_start_rotation - delta
            if self.rotation.x < 0: self.rotation.x += 2*math.pi
            elif self.rotation.x > 2*math.pi: self.rotation.x -= 2*math.pi
            self.rotation.y = min(max(self.rotation.y, -math.pi/2), math.pi/2)
        else:
            self.drag_start = None
            self.drag_start_rotation = None

        move = spy.float3(0,0,0)
        if input_state.is_down(spy.KeyCode.w): move += spy.float3(0,0,-1)
        if input_state.is_down(spy.KeyCode.s): move += spy.float3(0,0,1)
        if input_state.is_down(spy.KeyCode.a): move += spy.float3(-1,0,0)
        if input_state.is_down(spy.KeyCode.d): move += spy.float3(1,0,0)
        if input_state.is_down(spy.KeyCode.q): move += spy.float3(0,-1,0)
        if input_state.is_down(spy.KeyCode.e): move += spy.float3(0,1,0)
        
        move *= self.move_speed
        if input_state.is_down(spy.KeyCode.left_control): move *= 0.01
        if input_state.is_down(spy.KeyCode.left_shift): move *= 10.0

        self.position += spy.math.transform_vector(self.get_rotation(), move) * dt
        h = spy.math.length(self.position)
        if h < (1 + .01/6371): # limit to 10m above surface
            self.position = (1 + .01/6371) * self.position/h
