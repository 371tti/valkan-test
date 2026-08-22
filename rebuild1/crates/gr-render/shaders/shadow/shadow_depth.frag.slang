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

[shader("fragment")]
void main() {
    float alpha = frag_color.a * material.base_color_factor.a;
    if (alpha <= material.pbr_alpha.z) {
        discard;
    }
}
