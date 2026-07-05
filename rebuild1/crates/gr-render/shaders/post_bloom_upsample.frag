#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D source_image;

layout(push_constant) uniform BloomParams {
    vec4 source_texel;
    vec4 params;
} bloom;

vec3 bloom_tent_upsample(vec2 uv) {
    vec2 radius = bloom.source_texel.xy * max(bloom.params.z, 0.25);

    vec3 sum = texture(source_image, uv + vec2(-radius.x,  radius.y)).rgb;
    sum += texture(source_image, uv + vec2( 0.0,       radius.y)).rgb * 2.0;
    sum += texture(source_image, uv + vec2( radius.x,  radius.y)).rgb;
    sum += texture(source_image, uv + vec2(-radius.x,  0.0)).rgb * 2.0;
    sum += texture(source_image, uv).rgb * 4.0;
    sum += texture(source_image, uv + vec2( radius.x,  0.0)).rgb * 2.0;
    sum += texture(source_image, uv + vec2(-radius.x, -radius.y)).rgb;
    sum += texture(source_image, uv + vec2( 0.0,      -radius.y)).rgb * 2.0;
    sum += texture(source_image, uv + vec2( radius.x, -radius.y)).rgb;

    return max(sum * 0.0625, vec3(0.0));
}

void main() {
    out_color = vec4(bloom_tent_upsample(frag_uv), 1.0);
}
