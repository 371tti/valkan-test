#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 1) in vec3 frag_normal;
layout(location = 3) in vec4 frag_color;
layout(location = 5) in vec4 frag_shadow_pos[4];
layout(location = 9) in vec3 frag_world_pos;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_normal_roughness;
layout(location = 2) out vec4 out_transparent_normal_roughness;

layout(set = 0, binding = 0) uniform FrameCamera {
    mat4 view_proj;
    mat4 view;
    mat4 shadow_view_proj[4];
    vec4 shadow_cascade_splits;
    vec4 shadow_cascade_texel_world;
    vec4 shadow_cascade_depth_span;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
} frame_camera;

layout(set = 1, binding = 0) uniform MaterialParams {
    vec4 base_color_factor;
    vec4 emissive_occlusion;
    vec4 pbr_alpha;
    uvec4 flags;
} material;

layout(set = 2, binding = 0) uniform sampler2D shadow_cascade_0;
layout(set = 2, binding = 1) uniform sampler2D shadow_cascade_1;
layout(set = 2, binding = 2) uniform sampler2D shadow_cascade_2;
layout(set = 2, binding = 3) uniform sampler2D shadow_cascade_3;
layout(set = 2, binding = 4) uniform sampler2D translucent_shadow_0;
layout(set = 2, binding = 5) uniform sampler2D translucent_shadow_1;
layout(set = 2, binding = 6) uniform sampler2D translucent_shadow_2;
layout(set = 2, binding = 7) uniform sampler2D translucent_shadow_3;

#include "shadow_sampling.glsl"
#include "pbr_lighting.glsl"

vec4 apply_alpha(vec4 base_color) {
    if (material.flags.x == 1 && base_color.a <= material.pbr_alpha.z) {
        discard;
    }
    if (material.flags.x == 2 && base_color.a <= 0.001) {
        discard;
    }
    return base_color;
}

vec3 surface_normal() {
    vec3 normal = normalize_fast(frag_normal);
    if (material.flags.z != 0u && !gl_FrontFacing) {
        normal = -normal;
    }
    return normal;
}

void main() {
    vec4 base_color = apply_alpha(frag_color * material.base_color_factor);
    float metallic = clamp(material.pbr_alpha.x, 0.0, 1.0);
    float roughness = clamp(material.pbr_alpha.y, 0.04, 1.0);
    vec3 normal = surface_normal();
    vec4 material_meta = pack_view_normal_material(normal, roughness, metallic, base_color.rgb);
    out_color = vec4(
        shade_pbr(
            base_color.rgb,
            normal,
            metallic,
            roughness,
            1.0,
            material.emissive_occlusion.rgb,
            base_color.a
        ),
        base_color.a
    );
    if (material.flags.x == 2) {
        out_normal_roughness = vec4(0.0);
        out_transparent_normal_roughness = vec4(material_meta.xyz, 1.0 + gl_FragCoord.z);
    } else {
        out_normal_roughness = material_meta;
        out_transparent_normal_roughness = vec4(0.0);
    }
}
