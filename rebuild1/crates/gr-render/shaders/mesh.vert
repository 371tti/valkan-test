#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_tangent;
layout(location = 4) in vec4 in_color;

layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec2 frag_uv;
layout(location = 3) out vec4 frag_color;
layout(location = 4) out vec4 frag_tangent;
layout(location = 5) out vec4 frag_shadow_pos[4];
layout(location = 9) out vec3 frag_world_pos;

layout(set = 0, binding = 0) uniform FrameCamera {
    mat4 view_proj;
    mat4 view;
    mat4 shadow_view_proj[4];
    vec4 shadow_cascade_splits;
    vec4 shadow_cascade_texel_world;
    vec4 shadow_cascade_depth_span;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
    vec4 contact_shadow;
} frame_camera;

void main() {
    vec4 world_pos = vec4(in_position, 1.0);
    gl_Position = frame_camera.view_proj * world_pos;
    frag_normal = in_normal;
    frag_uv = in_uv;
    frag_color = in_color;
    frag_tangent = in_tangent;
    frag_shadow_pos[0] = frame_camera.shadow_view_proj[0] * world_pos;
    frag_shadow_pos[1] = frame_camera.shadow_view_proj[1] * world_pos;
    frag_shadow_pos[2] = frame_camera.shadow_view_proj[2] * world_pos;
    frag_shadow_pos[3] = frame_camera.shadow_view_proj[3] * world_pos;
    frag_world_pos = world_pos.xyz;
}
