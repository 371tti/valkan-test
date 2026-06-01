#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_color;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;

layout(set = 0, binding = 0) uniform FrameCamera {
    mat4 view_proj;
    mat4 shadow_view_proj;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
} frame_camera;

void main() {
    gl_Position = frame_camera.shadow_view_proj * vec4(in_position, 1.0);
    frag_uv = in_uv;
    frag_color = in_color;
}
