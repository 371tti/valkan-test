#version 450

layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in vec4 frag_color;
layout(location = 4) in vec4 frag_shadow_pos;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform FrameCamera {
    mat4 view_proj;
    mat4 shadow_view_proj;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
} frame_camera;

layout(set = 1, binding = 0) uniform MaterialParams {
    uint alpha_mode;
    float alpha_cutoff;
    uint has_base_color;
    uint _pad;
} material;

layout(set = 1, binding = 1) uniform sampler2D base_color_texture;
layout(set = 2, binding = 0) uniform sampler2D shadow_map;

vec4 apply_alpha(vec4 base_color) {
    if (material.alpha_mode == 1 && base_color.a <= material.alpha_cutoff) {
        discard;
    }
    return base_color;
}

float shadow_factor(float ndotl) {
    if (frag_shadow_pos.w <= 0.0) {
        return 1.0;
    }

    vec3 projected = frag_shadow_pos.xyz / frag_shadow_pos.w;
    vec2 uv = projected.xy * 0.5 + 0.5;

    if (
        uv.x < 0.0 || uv.x > 1.0 ||
        uv.y < 0.0 || uv.y > 1.0 ||
        projected.z < 0.0 || projected.z > 1.0
    ) {
        return 1.0;
    }

    float bias = max(0.0008 * (1.0 - ndotl), 0.00015);
    float compare = projected.z - bias;
    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
    float sum = 0.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float depth = texture(shadow_map, uv + vec2(x, y) * texel).r;
            sum += step(compare, depth);
        }
    }

    float lit = sum / 9.0;
    return mix(0.06, 1.0, lit);
}

vec3 shade(vec3 base_color) {
    vec3 normal = normalize(frag_normal);
    vec3 light = -frame_camera.light_dir.xyz;
    float ndotl = max(dot(normal, light), 0.0);
    float shadow = 1.0;
    if (ndotl > 0.0) {
        shadow = shadow_factor(ndotl);
    }
    vec3 ambient = base_color * frame_camera.ambient_color.rgb;
    vec3 direct = base_color * frame_camera.light_color.rgb * ndotl * shadow;
    return ambient + direct;
}

void main() {
    vec4 texture_color = texture(base_color_texture, frag_uv);
    vec4 base_color = apply_alpha(frag_color * texture_color);
    out_color = vec4(shade(base_color.rgb), base_color.a);
}
