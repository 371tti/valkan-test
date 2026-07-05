#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in vec4 frag_color;
layout(location = 4) in vec4 frag_tangent;
layout(location = 5) in vec4 frag_shadow_pos[4];
layout(location = 9) in vec3 frag_world_pos;
layout(location = 10) in float frag_view_depth;
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
    vec4 contact_shadow;
    mat4 local_shadow_view_proj[24];
    vec4 local_shadow_params[4];
    vec4 emissive_light_position_radius[4];
    vec4 emissive_light_color[4];
    vec4 emissive_light_direction_radius[4];
    vec4 emissive_light_size_kind[4];
    vec4 emissive_light_count;
} frame_camera;

layout(set = 1, binding = 0) uniform MaterialParams {
    vec4 base_color_factor;
    vec4 emissive_occlusion;
    vec4 pbr_alpha;
    uvec4 flags;
} material;

layout(set = 1, binding = 1) uniform sampler2D base_color_texture;
layout(set = 1, binding = 2) uniform sampler2D normal_texture;
layout(set = 1, binding = 3) uniform sampler2D metallic_roughness_texture;
layout(set = 1, binding = 4) uniform sampler2D occlusion_texture;
layout(set = 1, binding = 5) uniform sampler2D emissive_texture;
layout(set = 2, binding = 0) uniform sampler2D shadow_cascade_0;
layout(set = 2, binding = 1) uniform sampler2D shadow_cascade_1;
layout(set = 2, binding = 2) uniform sampler2D shadow_cascade_2;
layout(set = 2, binding = 3) uniform sampler2D shadow_cascade_3;
layout(set = 2, binding = 4) uniform sampler2D translucent_shadow_0;
layout(set = 2, binding = 5) uniform sampler2D translucent_shadow_1;
layout(set = 2, binding = 6) uniform sampler2D translucent_shadow_2;
layout(set = 2, binding = 7) uniform sampler2D translucent_shadow_3;
layout(set = 2, binding = 8) uniform sampler2D raw_shadow_cascade_0;
layout(set = 2, binding = 9) uniform sampler2D raw_shadow_cascade_1;
layout(set = 2, binding = 10) uniform sampler2D raw_shadow_cascade_2;
layout(set = 2, binding = 11) uniform sampler2D raw_shadow_cascade_3;
layout(set = 2, binding = 12) uniform samplerCube local_shadow_depth[4];

const uint TEX_BASE_COLOR = 1u << 0;
const uint TEX_NORMAL = 1u << 1;
const uint TEX_METALLIC_ROUGHNESS = 1u << 2;
const uint TEX_OCCLUSION = 1u << 3;
const uint TEX_EMISSIVE = 1u << 4;
#include "shadow_sampling.glsl"
#include "pbr_lighting.glsl"

bool has_texture(uint bit) {
    return (material.flags.y & bit) != 0u;
}

vec4 apply_alpha(vec4 base_color) {
    if (material.flags.x == 1 && base_color.a <= material.pbr_alpha.z) {
        discard;
    }
    if (material.flags.x == 2 && base_color.a <= 0.001) {
        discard;
    }
    return base_color;
}

vec3 normal_from_derivatives(vec3 vertex_normal, vec3 tangent_space_normal) {
    vec3 dp1 = dFdx(frag_world_pos);
    vec3 dp2 = dFdy(frag_world_pos);
    vec2 duv1 = dFdx(frag_uv);
    vec2 duv2 = dFdy(frag_uv);

    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) < 0.000001) {
        return vertex_normal;
    }

    float inv_det = 1.0 / det;
    vec3 tangent_raw = (dp1 * duv2.y - dp2 * duv1.y) * inv_det;
    if (dot(tangent_raw, tangent_raw) < 0.000001) {
        return vertex_normal;
    }

    vec3 tangent = tangent_raw - vertex_normal * dot(vertex_normal, tangent_raw);
    tangent = normalize_fast(tangent);

    vec3 bitangent = cross(vertex_normal, tangent);
    vec3 bitangent_raw = (-dp1 * duv2.x + dp2 * duv1.x) * inv_det;
    if (dot(bitangent, bitangent_raw) < 0.0) {
        bitangent = -bitangent;
    }

    return normalize_fast(mat3(tangent, bitangent, vertex_normal) * tangent_space_normal);
}

vec3 normal_from_map(vec3 vertex_normal, vec4 vertex_tangent) {
    vec3 map = texture(normal_texture, frag_uv).xyz * 2.0 - 1.0;
    map.xy *= material.pbr_alpha.w;
    vec3 tangent_space_normal = normalize_fast(map);

    vec3 tangent = vertex_tangent.xyz;
    float tangent_len_sq = dot(tangent, tangent);
    if (tangent_len_sq < 0.000001) {
        return normal_from_derivatives(vertex_normal, tangent_space_normal);
    }

    tangent = tangent - vertex_normal * dot(vertex_normal, tangent);
    tangent_len_sq = dot(tangent, tangent);
    if (tangent_len_sq < 0.000001) {
        return normal_from_derivatives(vertex_normal, tangent_space_normal);
    }
    tangent *= inversesqrt(tangent_len_sq);

    float tangent_sign = vertex_tangent.w < 0.0 ? -1.0 : 1.0;
    vec3 bitangent = cross(vertex_normal, tangent) * tangent_sign;
    return normalize_fast(mat3(tangent, bitangent, vertex_normal) * tangent_space_normal);
}

vec3 surface_normal() {
    vec3 normal = normalize_fast(frag_normal);
    if (material.flags.z != 0u && !gl_FrontFacing) {
        normal = -normal;
    }
    if (has_texture(TEX_NORMAL)) {
        normal = normal_from_map(normal, frag_tangent);
    }
    return normal;
}

vec4 read_base_color() {
    vec4 base_color = frag_color * material.base_color_factor;
    if (has_texture(TEX_BASE_COLOR)) {
        base_color *= texture(base_color_texture, frag_uv);
    }
    return apply_alpha(base_color);
}

void read_pbr(out float metallic, out float roughness) {
    metallic = clamp(material.pbr_alpha.x, 0.0, 1.0);
    roughness = clamp(material.pbr_alpha.y, 0.04, 1.0);
    if (has_texture(TEX_METALLIC_ROUGHNESS)) {
        vec4 mr = texture(metallic_roughness_texture, frag_uv);
        roughness = clamp(roughness * mr.g, 0.04, 1.0);
        metallic = clamp(metallic * mr.b, 0.0, 1.0);
    }
}

float read_occlusion() {
    if (!has_texture(TEX_OCCLUSION)) {
        return 1.0;
    }
    float strength = clamp(material.emissive_occlusion.a, 0.0, 1.0);
    if (strength <= 0.0) {
        return 1.0;
    }
    return mix(1.0, texture(occlusion_texture, frag_uv).r, strength);
}

vec3 read_emissive() {
    vec3 emissive = material.emissive_occlusion.rgb;
    if (dot(emissive, emissive) <= 0.0) {
        return vec3(0.0);
    }
    if (has_texture(TEX_EMISSIVE)) {
        emissive *= texture(emissive_texture, frag_uv).rgb;
    }
    return emissive;
}

void main() {
    vec4 base_color = read_base_color();
    float metallic;
    float roughness;
    read_pbr(metallic, roughness);
    vec3 normal = surface_normal();
    vec4 material_meta = pack_view_normal_material(normal, roughness, metallic, base_color.rgb);
    out_color = vec4(
        shade_pbr(
            base_color.rgb,
            normal,
            metallic,
            roughness,
            read_occlusion(),
            read_emissive(),
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
