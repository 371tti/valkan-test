#version 450
#extension GL_GOOGLE_include_directive : require

#include "scene_uniforms.glsl"

layout(set = 1, binding = 0) uniform sampler2D base_color_texture;

layout(location = 1) in vec2 frag_uv;

void main() {
    float alpha = object.base_color.a;
    if (object.texture_flags.x > 0.5) {
        alpha *= texture(base_color_texture, frag_uv).a;
    }

    if (alpha <= max(object.texture_info.w, 0.01)) {
        discard;
    }
}
