#version 450

layout(location = 0) in vec3 in_position;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_color;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;

layout(set = 0, binding = 0) uniform FrameCamera {
    mat4 view_proj;
    mat4 view;
    mat4 shadow_view_proj[3];
    vec4 shadow_cascade_splits;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
} frame_camera;

layout(push_constant) uniform ShadowCascade {
    uint cascade_index;
} shadow_cascade;

void main() {
    uint index = min(shadow_cascade.cascade_index, 2u);
    gl_Position = frame_camera.shadow_view_proj[index] * vec4(in_position, 1.0);
    frag_uv = in_uv;
    frag_color = in_color;
}
