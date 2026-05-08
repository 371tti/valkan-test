#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(set = 0, binding = 0) uniform Scene {
    mat4 view_proj;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
    vec4 camera_pos;
} scene;

layout(push_constant) uniform Object {
    mat4 model;
    vec4 base_color;
} object;

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec2 frag_uv;
layout(location = 2) out vec3 frag_world_pos;
layout(location = 3) out vec4 frag_base_color;

void main() {
    vec4 world_pos = object.model * vec4(in_position, 1.0);

    gl_Position = scene.view_proj * world_pos;
    frag_normal = mat3(object.model) * in_normal;
    frag_uv = in_uv;
    frag_world_pos = world_pos.xyz;
    frag_base_color = object.base_color;
}
