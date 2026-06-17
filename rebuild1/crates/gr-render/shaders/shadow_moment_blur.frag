#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_moments;

layout(set = 0, binding = 0) uniform sampler2D source_moments;

layout(push_constant) uniform BlurPush {
    vec2 texel_step;
    float radius_scale;
    float _pad;
} pc;

const float W0 = 0.20416400;
const float W1 = 0.18017382;
const float W2 = 0.12383154;
const float W3 = 0.06628225;
const float W4 = 0.02763055;
const float W12 = W1 + W2;
const float W34 = W3 + W4;
const float O12 = (W1 + W2 * 2.0) / W12;
const float O34 = (W3 * 3.0 + W4 * 4.0) / W34;

vec4 sample_moments(vec2 uv) {
    return texture(source_moments, uv);
}

void main() {
    vec2 step_uv = pc.texel_step * pc.radius_scale;
    vec4 moments = sample_moments(frag_uv) * W0;

    moments += (sample_moments(frag_uv + step_uv * O12) +
                sample_moments(frag_uv - step_uv * O12)) * W12;
    moments += (sample_moments(frag_uv + step_uv * O34) +
                sample_moments(frag_uv - step_uv * O34)) * W34;

    out_moments = clamp(moments, vec4(0.0), vec4(1.0));
}
