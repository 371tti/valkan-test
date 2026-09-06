#include "slang_compat.slang"
#include "common_math.slang"
#include "post_smaa.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_color;

uint smaa_debug_view() {
    return uint(params.debug.x + 0.5);
}

[shader("fragment")]
void main() {
    uint debug_view = smaa_debug_view();
    if (debug_view == 5u) {
        float2 edges = smaa_edge_sample(frag_uv);
        out_color = float4(edges.x, edges.y, 0.0, 1.0);
        return;
    }
    if (debug_view == 6u) {
        float4 weights = sample_texture(weights_tex, smaa_clamp_uv(frag_uv));
        float horizontal = max(weights.x, weights.z);
        float vertical = max(weights.y, weights.w);
        out_color = float4(horizontal, vertical, saturate(horizontal + vertical), 1.0);
        return;
    }
    out_color = float4(smaa_neighborhood_color(frag_uv, smaa_enabled()), 1.0);
}
