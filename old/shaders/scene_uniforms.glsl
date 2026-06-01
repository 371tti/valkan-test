#ifndef SCENE_UNIFORMS_GLSL
#define SCENE_UNIFORMS_GLSL

layout(set = 0, binding = 0) uniform Scene {
    mat4 view_proj;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
    vec4 camera_pos;
    vec4 reflection_probe_pos_radius;
    vec4 reflection_probe_box_min;
    vec4 reflection_probe_box_max;
    mat4 planar_view_proj;
    vec4 reflection_params;
    vec4 planar_plane;
    vec4 planar_params;
    vec4 planar_texture_info;
    mat4 shadow_view_proj;
    vec4 shadow_params;
    vec4 debug_params;
    vec4 camera_response;
    vec4 white_balance;
    vec4 gi_probe_pos_radius;
    vec4 gi_params;
    vec4 gi_sh[9];
    vec4 camera_basis_x;
    vec4 camera_basis_y;
    vec4 camera_basis_z;
    vec4 post_params;
} scene;

layout(push_constant) uniform Object {
    mat4 model;
    vec4 base_color;
    vec4 emissive_color;
    vec4 material;
    vec4 material_ext;
    vec4 texture_flags;
    vec4 texture_info;
} object;

int object_material_flags() {
    return int(object.texture_info.x + 0.5);
}

bool object_has_emissive_texture() {
    return (object_material_flags() & 1) != 0;
}

bool object_is_alpha_blend() {
    return (object_material_flags() & 2) != 0;
}

#endif
