#version 450
#extension GL_GOOGLE_include_directive : require

#include "scene_uniforms.glsl"

layout(set = 1, binding = 0) uniform sampler2D base_color_texture;

layout(location = 1) in vec2 frag_uv;

float bayer4(vec2 pixel) {
    ivec2 p = ivec2(floor(pixel)) & 3;
    int index = p.x + p.y * 4;
    const float thresholds[16] = float[](
        0.03125, 0.53125, 0.15625, 0.65625,
        0.78125, 0.28125, 0.90625, 0.40625,
        0.21875, 0.71875, 0.09375, 0.59375,
        0.96875, 0.46875, 0.84375, 0.34375
    );
    return thresholds[index];
}

void main() {
    float alpha = object.base_color.a;
    if (object.texture_flags.x > 0.5) {
        alpha *= texture(base_color_texture, frag_uv).a;
    }

    bool alpha_masked = !object_is_alpha_blend() && object.texture_info.w > 0.0;
    float cutoff = alpha_masked ? object.texture_info.w : 0.003;
    if (alpha <= cutoff) {
        discard;
    }

    float transmission = clamp(object.emissive_color.w, 0.0, 1.0);
    float opacity = alpha * mix(1.0, 0.18, transmission);
    if (opacity <= 0.02) {
        discard;
    }

    bool coverage_shadow = object_is_alpha_blend() || transmission > 0.05 || (!alpha_masked && alpha < 0.999);
    if (coverage_shadow && opacity <= bayer4(gl_FragCoord.xy)) {
        discard;
    }
}
