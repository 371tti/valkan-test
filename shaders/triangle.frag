#version 450
#extension GL_GOOGLE_include_directive : require

#define DEBUG_DEPTH 2.0
#define DEBUG_NORMALS 3.0
#define DEBUG_SHADOW_MASK 4.0
#define DEBUG_NO_TEXTURE 5.0

#include "scene_uniforms.glsl"

layout(set = 0, binding = 1) uniform samplerCube reflection_probe;
layout(set = 0, binding = 2) uniform sampler2D planar_reflection;
layout(set = 0, binding = 3) uniform sampler2D shadow_map;

layout(set = 1, binding = 0) uniform sampler2D base_color_texture;
layout(set = 1, binding = 1) uniform sampler2D metallic_roughness_texture;
layout(set = 1, binding = 2) uniform sampler2D normal_texture;
layout(set = 1, binding = 3) uniform sampler2D occlusion_texture;
layout(set = 1, binding = 4) uniform sampler2D emissive_texture;

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;
layout(location = 2) in vec3 frag_world_pos;
layout(location = 3) in vec4 frag_base_color;
layout(location = 0) out vec4 out_color;

#include "common.glsl"
#include "lit.glsl"

void main() {
    float debug_mode = scene.debug_params.x;
    bool use_textures = debug_mode != DEBUG_NO_TEXTURE;
    MaterialSample material = read_material(use_textures);

    if (scene.planar_texture_info.w > 0.5) {
        float plane_side = dot(frag_world_pos, normalize(scene.planar_plane.xyz)) + scene.planar_plane.w;
        if (plane_side <= scene.planar_texture_info.y) {
            discard;
        }
    }

    if (material.base.a < object.texture_info.w) {
        discard;
    }

    if (debug_mode == DEBUG_DEPTH) {
        float depth = linear_depth(gl_FragCoord.z);
        float near_plane = scene.debug_params.y;
        float far_plane = max(scene.debug_params.z, near_plane + 0.001);
        float visible_depth = 1.0 - saturate(log2(depth / near_plane + 1.0) / log2(far_plane / near_plane + 1.0));
        out_color = vec4(vec3(visible_depth), material.base.a);
        return;
    }
    if (debug_mode == DEBUG_NORMALS) {
        out_color = vec4(material.normal * 0.5 + 0.5, material.base.a);
        return;
    }
    if (debug_mode == DEBUG_SHADOW_MASK) {
        vec3 sun_light = normalize(-scene.light_dir.xyz);
        float visibility = shadow_visibility(frag_world_pos, sun_light, material.normal);
        out_color = vec4(vec3(visibility), material.base.a);
        return;
    }

    bool sample_reflection_probe = scene.reflection_params.w > 0.5;
    if (!sample_reflection_probe) {
        simplify_material_without_probe(material);
    }

    float view_dot = max(dot(material.normal, material.view), 0.0);
    vec3 f0 = material_f0(material);
    vec3 reflection = reflect(-material.view, material.normal);
    vec3 env_fresnel = fresnel_schlick(view_dot, f0);
    vec3 sun_light = normalize(-scene.light_dir.xyz);
    float sun_visibility = shadow_visibility(frag_world_pos, sun_light, material.normal);
    vec3 direct = material_direct_light(
        material,
        sun_light,
        scene.light_color.rgb * sun_visibility
    );
    GiSample gi = sample_gi(material, f0, view_dot, reflection, frag_world_pos);
    float reflection_strength = mix(0.35, 1.0, material.specular) * mix(1.0, 0.42, material.roughness);
    vec4 planar = planar_reflection_sample(frag_world_pos, material.normal, material.roughness);
    vec3 planar_specular =
        planar.rgb * env_fresnel * material.ao * reflection_strength * 1.1;
    float planar_mix = planar.a * (1.0 - 0.35 * material.roughness);
    vec3 specular_indirect = mix(gi.specular, planar_specular, saturate(planar_mix));

    vec3 color = gi.diffuse + direct + specular_indirect + material.emissive;
    color = apply_camera_response(color);

    out_color = vec4(color, material.base.a);
}
