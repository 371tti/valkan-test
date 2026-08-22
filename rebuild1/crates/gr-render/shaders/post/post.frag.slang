#include "slang_compat.slang"
#include "post_layout.slang"

#include "post_common.slang"
#include "post_fxaa.slang"
#include "post_ssao.slang"
#include "post_ssr.slang"
#include "post_bloom.slang"
#include "post_god_rays.slang"

[shader("fragment")]
void main() {
    float3 debug_color;
    if (post_debug_color(frag_uv, debug_color)) {
        out_color = float4(debug_color, 1.0);
        return;
    }
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
        float4 god_rays = post_god_rays(frag_uv);
        if (post_volumetric_fog_enabled()) {
            color = color * god_rays.a + god_rays.rgb;
        } else {
            color += god_rays.rgb;
        }
    }

    out_color = float4(apply_camera_effects(color), 1.0);
}
