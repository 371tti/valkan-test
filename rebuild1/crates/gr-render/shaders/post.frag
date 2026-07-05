#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 2, binding = 0) uniform sampler2D scene_color;
layout(set = 2, binding = 1) uniform sampler2D scene_depth;
layout(set = 2, binding = 2) uniform sampler2D scene_normal_roughness;
layout(set = 2, binding = 3) uniform sampler2D scene_transparent_normal_roughness;
layout(set = 2, binding = 4) uniform sampler2D bloom_texture;
layout(set = 2, binding = 5) uniform sampler2D god_ray_texture_0;
layout(set = 2, binding = 6) uniform sampler2D god_ray_texture_1;

layout(push_constant) uniform PostParams {
    vec4 white_balance;
    vec4 camera;
    vec4 depth;
    vec4 ssao;
    vec4 ssr;
    vec4 aa;
    vec4 shadow;
    vec4 bloom;
    vec4 features;
    vec4 god_ray_source0;
    vec4 god_ray_color0;
    vec4 god_ray_source1;
    vec4 god_ray_color1;
    float exposure;
    float contrast;
    float saturation;
    float enabled;
} params;

#include "post_common.glsl"
#include "post_fxaa.glsl"
#include "post_ssao.glsl"
#include "post_ssr.glsl"
#include "post_bloom.glsl"
#include "post_god_rays.glsl"

void main() {
    vec3 color = texture(scene_color, frag_uv).rgb;
    bool aa_enabled = post_aa_enabled();
    bool ssao_enabled = post_ssao_enabled();
    bool ssr_enabled = post_ssr_enabled();
    bool bloom_enabled = post_bloom_enabled();
    bool material_required = ssao_enabled || ssr_enabled;

    if (!aa_enabled && !material_required && !bloom_enabled) {
        out_color = vec4(apply_camera_effects(color), 1.0);
        return;
    }

    bool material_valid = material_required;
    SurfaceMaterial material = empty_surface_material(POST_BACKGROUND_DEPTH);
    if (material_valid) {
        material = surface_material(frag_uv);
    }

    color = high_quality_fxaa_scene_color(
        frag_uv,
        color,
        material,
        material_valid,
        aa_enabled
    );

    float ao = 1.0;

    if (ssao_enabled) {
        ao = screen_space_ao(frag_uv, material);

        float ao_strength = clamp(params.ssao.x, 0.0, 1.0) *
            0.32 *
            material.occlusion_weight;

        color *= mix(
            1.0 - ao_strength,
            1.0,
            ao
        );
    }

    if (ssr_enabled) {
        vec4 reflection = screen_space_reflection(frag_uv, material, ao);
        color = apply_screen_space_reflection(color, reflection);
    }

    if (bloom_enabled) {
        color += post_bloom(frag_uv);
        color += post_god_rays(frag_uv);
    }

    out_color = vec4(apply_camera_effects(color), 1.0);
}
