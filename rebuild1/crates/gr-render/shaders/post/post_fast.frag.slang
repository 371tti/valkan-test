#include "slang_compat.slang"
#include "post_layout.slang"
#include "post_camera_effects.slang"
#include "post_common.slang"

[shader("fragment")]
void main() {
    float3 debug_color;
    if (post_debug_color(frag_uv, debug_color)) {
        out_color = float4(debug_color, 1.0);
        return;
    }
    float3 color = sample_texture(scene_color, frag_uv).rgb;
    out_color = float4(apply_camera_effects(color), 1.0);
}
