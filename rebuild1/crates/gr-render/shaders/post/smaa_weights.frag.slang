#include "slang_compat.slang"
#include "common_math.slang"
#include "post_smaa.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_color;

[shader("fragment")]
void main() {
    if (!smaa_enabled()) {
        out_color = float4(0.0);
        return;
    }
    out_color = smaa_calculate_weights(frag_uv);
}
