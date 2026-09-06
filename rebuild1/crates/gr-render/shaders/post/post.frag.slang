#include "slang_compat.slang"
#include "post_layout.slang"

#include "post_common.slang"
#include "post_ssao.slang"
#include "post_ssr.slang"
#include "post_bloom.slang"
#include "post_god_rays.slang"

// Evaluates the complete post result at an arbitrary screen coordinate. The result is written to a
// dedicated post-color target; the following SMAA pass samples that target so its neighborhood is
// made from the exact final composition rather than re-running a partial set of effects.
float3 post_color_at(float2 uv) {
    uv = clamp_screen_uv(uv);
    float3 color = sample_texture(scene_color, uv).rgb;
    bool ssao_enabled = post_ssao_enabled();
    bool ssr_enabled = post_ssr_enabled();
    bool bloom_enabled = post_bloom_enabled();
    bool god_rays_enabled = post_god_rays_enabled();
    bool material_required = ssao_enabled || ssr_enabled;
    bool material_valid = material_required;
    SurfaceMaterial material = empty_surface_material(POST_BACKGROUND_DEPTH);

    if (material_valid) {
        material = surface_material(uv);
    }

    float ao = 1.0;
    if (ssao_enabled) {
        ao = screen_space_ao(uv, material);
        float ao_strength = clamp(params.ssao.x, 0.0, 1.0) *
            0.32 * material.occlusion_weight;
        color *= lerp(1.0 - ao_strength, 1.0, ao);
    }

    if (ssr_enabled) {
        float4 reflection = screen_space_reflection(uv, material, ao);
        color = apply_screen_space_reflection(color, reflection);
    }

    if (bloom_enabled) {
        color += post_bloom(uv);
    }

    if (god_rays_enabled) {
        float4 god_rays = post_god_rays(uv);
        if (post_volumetric_fog_enabled()) {
            // The dedicated camera-ray pass also carries medium transmittance in alpha.  Do not
            // apply that extinction to the whole opaque scene when the ray contains no visible
            // scattering: in a shadowed view the direct term is nearly zero, so the raw alpha
            // would turn the entire frame black even though this pass is only meant to add the
            // visible shaft/fog contribution.  Restrict surface attenuation to pixels where the
            // resolved volume has a meaningful radiance signal.
            float scattering_luma = luminance_of(max(god_rays.rgb, float3(0.0)));
            float medium_signal = smoothstep(0.012, 0.10, scattering_luma);
            float surface_transmittance = lerp(1.0, saturate(god_rays.a), medium_signal);
            color = color * surface_transmittance + god_rays.rgb;
        } else {
            color += god_rays.rgb;
        }
    }

    return apply_camera_effects(color);
}

[shader("fragment")]
void main() {
    if (post_debug_view() == 7u) {
        float visibility = screen_space_ao_debug(frag_uv);
        out_color = float4(visibility.xxx, 1.0);
        return;
    }
    float3 debug_color;
    if (post_debug_color(frag_uv, debug_color)) {
        out_color = float4(debug_color, 1.0);
        return;
    }
    out_color = float4(post_color_at(frag_uv), 1.0);
}
