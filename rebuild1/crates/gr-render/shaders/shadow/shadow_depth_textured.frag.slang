#include "slang_compat.slang"

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

[shader("fragment")]
void main() {
    // Directional/local depth passes only need alpha; avoid carrying RGB through the alpha test.
    float alpha = frag_color.a
        * material.base_color_factor.a
        * sample_texture(base_color_texture, frag_uv).a;
    if (alpha <= material.pbr_alpha.z) {
        discard;
    }
}
