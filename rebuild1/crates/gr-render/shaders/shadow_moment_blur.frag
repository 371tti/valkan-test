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

vec4 sample_moments(vec2 uv) {
    return texture(source_moments, uv);
}

void main() {
    vec2 step_uv = pc.texel_step * pc.radius_scale;
    vec4 moments = sample_moments(frag_uv) * W0;

    moments += (sample_moments(frag_uv + step_uv) +
                sample_moments(frag_uv - step_uv)) * W1;
    moments += (sample_moments(frag_uv + step_uv * 2.0) +
                sample_moments(frag_uv - step_uv * 2.0)) * W2;
    moments += (sample_moments(frag_uv + step_uv * 3.0) +
                sample_moments(frag_uv - step_uv * 3.0)) * W3;
    moments += (sample_moments(frag_uv + step_uv * 4.0) +
                sample_moments(frag_uv - step_uv * 4.0)) * W4;

    out_moments = clamp(moments, vec4(0.0), vec4(1.0));
}
