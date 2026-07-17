#include "slang_compat.slang"

#include "shadow_alpha.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 1) in float4 frag_color;

struct MaterialParams {
    float4 base_color_factor;
    float4 emissive_occlusion;
    float4 pbr_alpha;
    uint4 flags;
};
[[vk::binding(0, 1)]] ConstantBuffer<MaterialParams> material;

[[vk::binding(1, 1)]] Sampler2D base_color_texture;

static const uint TEX_BASE_COLOR = 1u << 0;

bool has_base_color_texture() {
    return (material.flags.y & TEX_BASE_COLOR) != 0u;
}

[shader("fragment")]
void main() {
    float4 base_color = frag_color * material.base_color_factor;

    if (has_base_color_texture()) {
        base_color *= sample_texture(base_color_texture, frag_uv);
    }

    discard_opaque_shadow_alpha(
        material.flags.x,
        material.pbr_alpha.z,
        base_color.a
    );
}
