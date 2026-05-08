#version 450

layout(set = 0, binding = 0) uniform Scene {
    mat4 view_proj;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
    vec4 camera_pos;
} scene;

layout(set = 1, binding = 0) uniform sampler2D base_color_texture;
layout(set = 1, binding = 1) uniform sampler2D metallic_roughness_texture;
layout(set = 1, binding = 2) uniform sampler2D normal_texture;
layout(set = 1, binding = 3) uniform sampler2D occlusion_texture;
layout(set = 1, binding = 4) uniform sampler2D emissive_texture;

layout(push_constant) uniform Object {
    mat4 model;
    vec4 base_color;
    vec4 emissive_color;
    vec4 material;
    vec4 texture_flags;
    vec4 texture_info;
} object;

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;
layout(location = 2) in vec3 frag_world_pos;
layout(location = 3) in vec4 frag_base_color;
layout(location = 0) out vec4 out_color;

vec3 normal_from_map(vec3 vertex_normal, vec3 world_pos, vec2 uv) {
    vec3 map = texture(normal_texture, uv).xyz * 2.0 - 1.0;
    map.xy *= object.texture_info.y;

    vec3 dp1 = dFdx(world_pos);
    vec3 dp2 = dFdy(world_pos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 normal = normalize(vertex_normal);
    vec3 tangent = normalize(dp1 * duv2.y - dp2 * duv1.y);
    vec3 bitangent = normalize(-dp1 * duv2.x + dp2 * duv1.x);

    if (dot(tangent, tangent) < 0.0001 || dot(bitangent, bitangent) < 0.0001) {
        return normal;
    }

    return normalize(mat3(tangent, bitangent, normal) * normalize(map));
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

vec3 simple_environment(vec3 reflection) {
    float sky = clamp(reflection.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 floor_color = vec3(0.18, 0.16, 0.13);
    vec3 sky_color = vec3(0.48, 0.58, 0.72);

    return mix(floor_color, sky_color, sky);
}

void main() {
    vec3 normal = normalize(frag_normal);
    if (object.texture_flags.z > 0.5) {
        normal = normal_from_map(normal, frag_world_pos, frag_uv);
    }

    vec3 light = normalize(-scene.light_dir.xyz);
    vec3 view = normalize(scene.camera_pos.xyz - frag_world_pos);
    vec3 half_vec = normalize(light + view);
    vec4 texel = texture(base_color_texture, frag_uv);
    vec4 mr_texel = texture(metallic_roughness_texture, frag_uv);
    float metallic = clamp(object.material.x, 0.0, 1.0);
    float roughness = clamp(object.material.y, 0.04, 1.0);
    float specular_strength = clamp(object.material.z, 0.0, 1.0);
    float ao = clamp(object.material.w, 0.0, 1.0);
    vec3 emissive = object.emissive_color.rgb;

    if (object.texture_flags.y > 0.5) {
        roughness *= mr_texel.g;
        metallic *= mr_texel.b;
    }
    if (object.texture_flags.w > 0.5) {
        float occlusion = texture(occlusion_texture, frag_uv).r;
        ao *= mix(1.0, occlusion, object.texture_info.z);
    }
    if (object.texture_info.x > 0.5) {
        emissive *= texture(emissive_texture, frag_uv).rgb;
    }

    float diffuse = max(dot(normal, light), 0.0);
    vec4 base = frag_base_color * texel;
    if (object.texture_flags.x < 0.5) {
        base = frag_base_color;
    }

    float view_dot = max(dot(normal, view), 0.0);
    float half_dot = max(dot(normal, half_vec), 0.0);
    vec3 dielectric = vec3(0.04 * specular_strength);
    vec3 f0 = mix(dielectric, base.rgb, metallic);
    vec3 fresnel = fresnel_schlick(view_dot, f0);
    vec3 half_fresnel = fresnel_schlick(half_dot, f0);
    vec3 diffuse_color = base.rgb * (1.0 - metallic);
    float shininess = mix(128.0, 8.0, roughness);
    float specular = pow(half_dot, shininess) * specular_strength * mix(1.0, 0.35, roughness);
    vec3 reflection = reflect(-view, normal);
    vec3 environment = simple_environment(reflection) * scene.ambient.rgb * ao;
    vec3 environment_specular = fresnel * environment * mix(1.4, 0.35, roughness);

    if (base.a < object.texture_info.w) {
        discard;
    }

    vec3 color = diffuse_color * (scene.ambient.rgb * ao + scene.light_color.rgb * diffuse);
    color += half_fresnel * scene.light_color.rgb * specular;
    color += environment_specular;
    color += emissive;

    out_color = vec4(color, base.a);
}
