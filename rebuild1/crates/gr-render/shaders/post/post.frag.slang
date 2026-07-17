#include "slang_compat.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_color;

[[vk::binding(0, 2)]] Sampler2D scene_color;
[[vk::binding(1, 2)]] Sampler2D scene_depth;
[[vk::binding(2, 2)]] Sampler2D scene_normal_roughness;
[[vk::binding(3, 2)]] Sampler2D scene_transparent_normal_roughness;
[[vk::binding(4, 2)]] Sampler2D bloom_texture;
[[vk::binding(5, 2)]] Sampler2D god_ray_texture_0;
[[vk::binding(6, 2)]] Sampler2D god_ray_texture_1;

struct PostParams {
    float4 white_balance;
    float4 camera;
    float4 depth;
    float4 ssao;
    float4 ssr;
    float4 aa;
    float4 shadow;
    float4 bloom;
    float4 features;
    float4 god_ray_source0;
    float4 god_ray_color0;
    float4 god_ray_source1;
    float4 god_ray_color1;
    float exposure;
    float contrast;
    float saturation;
    float enabled;
};
[[vk::push_constant]] ConstantBuffer<PostParams> params;

#include "post_common.slang"
#include "post_fxaa.slang"
#include "post_ssao.slang"
#include "post_ssr.slang"
#include "post_bloom.slang"
#include "post_god_rays.slang"

[shader("fragment")]
void main() {
    float3 color = sample_texture(scene_color, frag_uv).rgb;
    bool aa_enabled = post_aa_enabled();
    bool ssao_enabled = post_ssao_enabled();
    bool ssr_enabled = post_ssr_enabled();
    bool bloom_enabled = post_bloom_enabled();
    bool god_rays_enabled = post_god_rays_enabled();
    bool material_required = ssao_enabled || ssr_enabled;
    bool post_light_enabled = bloom_enabled || god_rays_enabled;

    if (!aa_enabled && !material_required && !post_light_enabled) {
        out_color = float4(apply_camera_effects(color), 1.0);
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

        color *= lerp(
            1.0 - ao_strength,
            1.0,
            ao
        );
    }

    if (ssr_enabled) {
        float4 reflection = screen_space_reflection(frag_uv, material, ao);
        color = apply_screen_space_reflection(color, reflection);
    }

    if (bloom_enabled) {
        color += post_bloom(frag_uv);
    }

    if (god_rays_enabled) {
        color += post_god_rays(frag_uv);
    }

    out_color = float4(apply_camera_effects(color), 1.0);
}
