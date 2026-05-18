#version 450

layout(set = 1, binding = 0) uniform sampler2D base_color_texture;

layout(push_constant) uniform Object {
    mat4 model;
    vec4 base_color;
    vec4 emissive_color;
    vec4 material;
    vec4 texture_flags;
    vec4 texture_info;
} object;

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
