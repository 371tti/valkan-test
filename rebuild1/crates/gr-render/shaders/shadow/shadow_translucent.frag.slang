#include "slang_compat.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 1) in float4 frag_color;

layout(location = 0) out float4 out_transmittance;

struct MaterialParams {
    float4 base_color_factor;
    float4 emissive_occlusion;
    float4 pbr_alpha;
    uint4 flags;
};
[[vk::binding(0, 1)]] ConstantBuffer<MaterialParams> material;

[[vk::binding(0, 2)]] Sampler2D shadow_cascade_0;
[[vk::binding(1, 2)]] Sampler2D shadow_cascade_1;
[[vk::binding(2, 2)]] Sampler2D shadow_cascade_2;
[[vk::binding(3, 2)]] Sampler2D shadow_cascade_3;

struct ShadowCascade {
    uint cascade_index;
};
[[vk::push_constant]] ConstantBuffer<ShadowCascade> shadow_cascade;

#include "shadow_alpha.slang"

int2 shadow_texel_coord(float4 sv_position) {
    return int2(sv_position.xy);
}

float opaque_depth_for_cascade(uint index, int2 texel_coord) {
    index = min(index, 3u);

    if (index == 0u) {
        return load_texel(
            shadow_cascade_0,
            texel_coord,
            0
        ).r;
    }

    if (index == 1u) {
        return load_texel(
            shadow_cascade_1,
            texel_coord,
            0
        ).r;
    }

    if (index == 2u) {
        return load_texel(
            shadow_cascade_2,
            texel_coord,
            0
        ).r;
    }

    return load_texel(
        shadow_cascade_3,
        texel_coord,
        0
    ).r;
}

void discard_when_hidden_by_opaque(float4 sv_position) {
    uint index = min(shadow_cascade.cascade_index, 3u);
    float opaque_depth = opaque_depth_for_cascade(index, shadow_texel_coord(sv_position));

    if (sv_position.z > opaque_depth + 0.00065) {
        discard;
    }
}

float3 translucent_transmittance(float3 color, float alpha) {
    color = clamp(color, float3(0.0), float3(1.0));
    alpha = clamp(alpha, 0.0, 1.0);

    return lerp(float3(1.0), color, alpha);
}

[shader("fragment")]
void main(float4 sv_position : SV_Position) {
    float4 base_color = frag_color * material.base_color_factor;
    float alpha = clamp(base_color.a, 0.0, 1.0);

    discard_translucent_shadow_alpha(material.flags.x, alpha);
    discard_when_hidden_by_opaque(sv_position);

    out_transmittance = float4(
        translucent_transmittance(base_color.rgb, alpha),
        clamp(sv_position.z, 0.0, 1.0)
    );
}
