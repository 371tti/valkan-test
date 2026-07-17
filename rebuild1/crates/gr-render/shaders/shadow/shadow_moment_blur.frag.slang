#include "slang_compat.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float2 out_moments;

[[vk::binding(0, 0)]] Sampler2D source_moments;

struct BlurPush {
    float2 texel_step;
    float radius_scale;
    float _pad;
};
[[vk::push_constant]] ConstantBuffer<BlurPush> pc;

static const float W0 = 0.20416400;
static const float W1 = 0.18017382;
static const float W2 = 0.12383154;
static const float W3 = 0.06628225;
static const float W4 = 0.02763055;
static const float W12 = W1 + W2;
static const float W34 = W3 + W4;
static const float O12 = (W1 + W2 * 2.0) / W12;
static const float O34 = (W3 * 3.0 + W4 * 4.0) / W34;

float2 sample_moments(float2 uv) {
    return sample_texture(source_moments, uv).xy;
}

[shader("fragment")]
void main() {
    float2 step_uv = pc.texel_step * pc.radius_scale;
    float2 moments = sample_moments(frag_uv) * W0;

    moments += (sample_moments(frag_uv + step_uv * O12) +
                sample_moments(frag_uv - step_uv * O12)) * W12;
    moments += (sample_moments(frag_uv + step_uv * O34) +
                sample_moments(frag_uv - step_uv * O34)) * W34;

    out_moments = clamp(moments, float2(0.0), float2(1.0));
}
