#version 450
#extension GL_GOOGLE_include_directive : require

#include "scene_uniforms.glsl"

layout(set = 1, binding = 0) uniform sampler2D base_color_texture;

layout(location = 1) in vec2 frag_uv;
layout(location = 0) out vec4 out_opacity;

void main() {
    float alpha = object.base_color.a;
    if (object.texture_flags.x > 0.5) {
        alpha *= texture(base_color_texture, frag_uv).a;
    }

    if (alpha <= 0.01) {
        discard;
    }

    float transmission = clamp(object.emissive_color.w, 0.0, 1.0);
    float opacity = alpha * mix(1.0, 0.16, transmission);

    out_opacity = vec4(vec3(clamp(opacity, 0.0, 1.0)), 1.0);
}
