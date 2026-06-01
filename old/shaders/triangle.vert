#version 450
#extension GL_GOOGLE_include_directive : require

#include "scene_uniforms.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec2 frag_uv;
layout(location = 2) out vec3 frag_world_pos;
layout(location = 3) out vec4 frag_base_color;

void main() {
    vec4 world_pos = object.model * vec4(in_position, 1.0);
    mat3 normal_matrix = transpose(inverse(mat3(object.model)));

    gl_Position = scene.view_proj * world_pos;
    frag_normal = normal_matrix * in_normal;
    frag_uv = in_uv;
    frag_world_pos = world_pos.xyz;
    frag_base_color = object.base_color;
}
