#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 2, binding = 0) uniform sampler2D scene_color;
layout(set = 2, binding = 1) uniform sampler2D scene_depth;
layout(set = 2, binding = 2) uniform sampler2D scene_normal_roughness;
layout(set = 2, binding = 3) uniform sampler2D scene_transparent_normal_roughness;

layout(push_constant) uniform PostParams {
    vec4 white_balance;
    vec4 camera;
    vec4 depth;
    vec4 ssao;
    vec4 ssr;
    vec4 aa;
    float exposure;
    float contrast;
    float saturation;
    float enabled;
} params;

#include "post_common.glsl"
#include "post_fxaa.glsl"
#include "post_ssao.glsl"
#include "post_ssr.glsl"

void main() {
    bool ssao_enabled = params.ssao.x > 0.0;
    bool ssr_enabled = params.ssr.x > 0.0 && params.ssr.y >= 1.0;

    SurfaceMaterial material = SurfaceMaterial(
        vec3(0.0, 0.0, 1.0),
        1.0,
        1.0,
        0.0,
        0.0
    );
    bool material_valid = false;

    if (ssao_enabled || ssr_enabled) {
        material = surface_material(frag_uv);
        material_valid = true;
    }

    vec3 color = texture(scene_color, frag_uv).rgb;

    color = high_quality_fxaa_scene_color(
        frag_uv,
        color,
        material,
        material_valid
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

    out_color = vec4(apply_camera_effects(color), 1.0);
}
