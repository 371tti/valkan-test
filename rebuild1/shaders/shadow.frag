#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(set = 1, binding = 0) uniform MaterialParams {
    uint alpha_mode;
    float alpha_cutoff;
    uint has_base_color;
    uint _pad;
} material;

void main() {
    if (material.alpha_mode == 1 && frag_color.a <= material.alpha_cutoff) {
        discard;
    }
}
