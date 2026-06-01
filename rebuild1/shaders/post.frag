#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 2, binding = 0) uniform sampler2D scene_color;

layout(push_constant) uniform PostParams {
    vec4 white_balance;
    float exposure;
    float contrast;
    float saturation;
    float enabled;
} params;

vec3 aces_tonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

vec3 apply_camera_effects(vec3 color) {
    if (params.enabled < 0.5) {
        return max(color, vec3(0.0));
    }

    color = max(color, vec3(0.0)) * params.exposure * params.white_balance.rgb;
    color = aces_tonemap(color);
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luminance), color, params.saturation);
    color = mix(vec3(0.5), color, params.contrast);

    return clamp(color, 0.0, 1.0);
}

void main() {
    vec3 color = texture(scene_color, frag_uv).rgb;
    out_color = vec4(apply_camera_effects(color), 1.0);
}
