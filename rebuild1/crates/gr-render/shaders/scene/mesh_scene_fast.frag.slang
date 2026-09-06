#define SCENE_FAST_PATH
#include "slang_compat.slang"

layout(location = 1) in float3 frag_normal;
layout(location = 3) in float4 frag_color;
layout(location = 9) in float3 frag_world_pos;
layout(location = 10) in float frag_view_depth;
layout(location = 0) out float4 out_color;

struct FrameCamera {
    float4x4 view_proj;
    float4x4 view;
    float4x4 shadow_view_proj[4];
    float4 stable_csm_pcss_params;
    float4 stable_csm_receiver_params;
    float4 shadow_cascade_splits;
    float4 shadow_cascade_texel_world;
    float4 shadow_cascade_depth_span;
    float4 shadow_cascade_resolution;
    float4 camera_pos;
    float4 light_dir;
    float4 light_color;
    float4 ambient_color;
    float4x4 local_shadow_view_proj[24];
    float4 local_shadow_params[4];
    float4 emissive_light_position_radius[4];
    float4 emissive_light_color[4];
    float4 emissive_light_direction_radius[4];
    float4 emissive_light_size_kind[4];
    float4 emissive_light_count;
    float4 debug_view;
    float4x4 previous_view_projection;
    float4 pcss_temporal;
};
[[vk::binding(0, 0)]] ConstantBuffer<FrameCamera> frame_camera;
[[vk::binding(1, 0)]] Sampler2D pcss_shadow_history;

struct MaterialParams {
    float4 base_color_factor;
    float4 emissive_occlusion;
    float4 pbr_alpha;
    uint4 flags;
};
[[vk::binding(0, 1)]] ConstantBuffer<MaterialParams> material;

[[vk::binding(4, 2)]] Sampler2D translucent_shadow_0;
[[vk::binding(5, 2)]] Sampler2D translucent_shadow_1;
[[vk::binding(6, 2)]] Sampler2D translucent_shadow_2;
[[vk::binding(7, 2)]] Sampler2D translucent_shadow_3;
[[vk::binding(8, 2)]] SamplerCube local_shadow_depth[4];
[[vk::binding(9, 2)]] Sampler2DArrayShadow shadow_depth;
[[vk::binding(10, 2)]] Sampler2DArray shadow_depth_raw;

#include "stable_csm_sampling.slang"
#include "pbr_lighting.slang"

#ifdef SCENE_OPAQUE
float4 apply_alpha(float4 base_color) {
    return float4(base_color.rgb, 1.0);
}
#else
float4 apply_alpha(float4 base_color) {
    if (material.flags.x == 1 && base_color.a <= material.pbr_alpha.z) {
        discard;
    }
    if (material.flags.x == 2 && base_color.a <= 0.001) {
        discard;
    }
    return base_color;
}
#endif

float3 surface_geometric_normal(bool front_facing) {
    float3 normal = normalize_fast(frag_normal);
    if (material.flags.z != 0u && !front_facing) {
        normal = -normal;
    }
    return normal;
}

#ifdef SCENE_OPAQUE
[earlydepthstencil]
#endif
[shader("fragment")]
void main(float4 sv_position : SV_Position, bool front_facing : SV_IsFrontFace) {
    float4 base_color = apply_alpha(frag_color * material.base_color_factor);
    float3 normal = surface_geometric_normal(front_facing);
    if (scene_lighting_disabled() && frame_camera.debug_view.x < 2.5) {
        out_color = float4(material.emissive_occlusion.rgb, base_color.a);
        return;
    }
    float metallic = clamp(material.pbr_alpha.x, 0.0, 1.0);
    float roughness = clamp(material.pbr_alpha.y, 0.04, 1.0);
    float pcss_visibility;
    out_color = float4(
        shade_pbr(
            base_color.rgb,
            normal,
            normal,
            metallic,
            roughness,
            1.0,
            material.emissive_occlusion.rgb,
            base_color.a,
            sv_position.xy,
            pcss_visibility
        ),
        base_color.a
    );
}
