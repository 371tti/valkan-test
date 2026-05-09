#version 450

#define MAX_EMISSIVE_LIGHTS 8
#define PI 3.14159265359

layout(set = 0, binding = 0) uniform Scene {
    mat4 view_proj;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
    vec4 camera_pos;
    vec4 reflection_probe_pos_radius;
    vec4 reflection_probe_box_min;
    vec4 reflection_probe_box_max;
    vec4 point_light_count;
    vec4 point_light_pos_radius[MAX_EMISSIVE_LIGHTS];
    vec4 point_light_color_power[MAX_EMISSIVE_LIGHTS];
    mat4 planar_view_proj;
    vec4 reflection_params;
    vec4 planar_plane;
    vec4 planar_params;
    vec4 planar_texture_info;
} scene;
layout(set = 0, binding = 1) uniform samplerCube reflection_probe;
layout(set = 0, binding = 2) uniform sampler2D planar_reflection;

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

float distribution_ggx(vec3 normal, vec3 half_vec, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float ndoth = max(dot(normal, half_vec), 0.0);
    float ndoth2 = ndoth * ndoth;
    float denom = ndoth2 * (alpha2 - 1.0) + 1.0;

    return alpha2 / max(PI * denom * denom, 0.0001);
}

float geometry_schlick_ggx(float ndotv, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    return ndotv / max(ndotv * (1.0 - k) + k, 0.0001);
}

float geometry_smith(vec3 normal, vec3 view, vec3 light, float roughness) {
    float ndotv = max(dot(normal, view), 0.0);
    float ndotl = max(dot(normal, light), 0.0);

    return geometry_schlick_ggx(ndotv, roughness) * geometry_schlick_ggx(ndotl, roughness);
}

vec3 brdf_light(
    vec3 normal,
    vec3 view,
    vec3 light,
    vec3 radiance,
    vec3 albedo,
    float metallic,
    float roughness,
    float specular_strength
) {
    vec3 half_vec = normalize(view + light);
    float ndotv = max(dot(normal, view), 0.0);
    float ndotl = max(dot(normal, light), 0.0);
    float vdoth = max(dot(view, half_vec), 0.0);
    vec3 f0 = mix(vec3(0.04 * specular_strength), albedo, metallic);
    vec3 fresnel = fresnel_schlick(vdoth, f0);
    float distribution = distribution_ggx(normal, half_vec, roughness);
    float geometry = geometry_smith(normal, view, light, roughness);
    vec3 specular = distribution * geometry * fresnel / max(4.0 * ndotv * ndotl, 0.0001);
    vec3 diffuse = (1.0 - fresnel) * (1.0 - metallic) * albedo / PI;

    return (diffuse + specular) * radiance * ndotl;
}

vec3 environment_color(vec3 direction) {
    direction = normalize(direction);
    float up = clamp(direction.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ground = mix(vec3(0.055, 0.048, 0.042), vec3(0.38, 0.31, 0.23), clamp(-direction.y * 0.7 + 0.2, 0.0, 1.0));
    vec3 sky = mix(vec3(0.06, 0.10, 0.17), vec3(0.72, 0.82, 1.0), pow(up, 0.65));
    vec3 horizon = vec3(1.0, 0.78, 0.48) * exp(-abs(direction.y) * 9.0) * 0.28;
    vec3 env = mix(ground, sky, smoothstep(-0.08, 0.18, direction.y)) + horizon;
    vec3 sun_dir = normalize(-scene.light_dir.xyz);
    float sun = pow(max(dot(direction, sun_dir), 0.0), 350.0);

    return env + scene.light_color.rgb * sun * 8.0;
}

vec3 safe_direction(vec3 direction) {
    return vec3(
        abs(direction.x) < 0.0001 ? (direction.x < 0.0 ? -0.0001 : 0.0001) : direction.x,
        abs(direction.y) < 0.0001 ? (direction.y < 0.0 ? -0.0001 : 0.0001) : direction.y,
        abs(direction.z) < 0.0001 ? (direction.z < 0.0 ? -0.0001 : 0.0001) : direction.z
    );
}

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

vec2 clamp_uv(vec2 uv) {
    return clamp(uv, vec2(0.0), vec2(1.0));
}

vec3 parallax_correct_reflection(vec3 world_pos, vec3 reflection) {
    if (scene.reflection_probe_pos_radius.w <= 0.0 || scene.reflection_params.y < 0.5) {
        return reflection;
    }

    vec3 direction = safe_direction(normalize(reflection));
    vec3 box_min = scene.reflection_probe_box_min.xyz;
    vec3 box_max = scene.reflection_probe_box_max.xyz;
    vec3 first = (box_min - world_pos) / direction;
    vec3 second = (box_max - world_pos) / direction;
    vec3 far_plane = max(first, second);
    float distance = min(min(far_plane.x, far_plane.y), far_plane.z);

    if (distance <= 0.0) {
        return reflection;
    }

    vec3 hit = world_pos + direction * distance;
    vec3 center_to_world = world_pos - scene.reflection_probe_pos_radius.xyz;
    float center_distance = length(center_to_world);
    float probe_radius = max(scene.reflection_probe_pos_radius.w, 0.001);
    float interior = clamp(center_distance / probe_radius, 0.0, 1.0);
    float parallax_weight = smoothstep(0.05, 0.85, interior);
    vec3 corrected = normalize(hit - scene.reflection_probe_pos_radius.xyz);

    return normalize(mix(reflection, corrected, parallax_weight));
}

vec3 environment_reflection(vec3 reflection, float roughness, vec3 world_pos) {
    if (scene.camera_pos.w > 0.5 && scene.reflection_params.w > 0.5) {
        vec3 probe_direction = parallax_correct_reflection(world_pos, reflection);
        vec3 probe = texture(reflection_probe, probe_direction).rgb;
        vec3 fallback = environment_color(reflection);
        float probe_radius = max(scene.reflection_probe_pos_radius.w, 0.001);
        float world_to_center = length(world_pos - scene.reflection_probe_pos_radius.xyz);
        float probe_coverage = 1.0 - smoothstep(probe_radius * 0.65, probe_radius * 1.35, world_to_center);
        float roughness_fade = roughness * roughness * scene.reflection_params.z;
        vec3 reflected = mix(probe, fallback, roughness_fade);

        return mix(fallback, reflected, scene.reflection_params.x * probe_coverage);
    }

    vec3 sharp = environment_color(reflection);
    vec3 soft = (
        environment_color(normalize(reflection + vec3(0.45, 0.25, 0.12))) +
        environment_color(normalize(reflection + vec3(-0.38, 0.18, -0.16))) +
        environment_color(normalize(reflection + vec3(0.0, 0.5, 0.0))) +
        environment_color(normalize(reflection + vec3(0.0, -0.35, 0.0)))
    ) * 0.25;
    float blur = roughness * roughness;

    return mix(sharp, soft, blur);
}

vec3 planar_reflection_color(vec2 uv, float roughness) {
    vec2 clamped_uv = clamp_uv(uv);
    vec2 texel = 1.0 / vec2(textureSize(planar_reflection, 0));
    float blur = roughness * roughness;
    vec2 axis = texel * mix(1.0, 6.0, blur);

    vec3 center = texture(planar_reflection, clamped_uv).rgb;
    vec3 cross = (
        texture(planar_reflection, clamp_uv(clamped_uv + vec2(axis.x, 0.0))).rgb +
        texture(planar_reflection, clamp_uv(clamped_uv - vec2(axis.x, 0.0))).rgb +
        texture(planar_reflection, clamp_uv(clamped_uv + vec2(0.0, axis.y))).rgb +
        texture(planar_reflection, clamp_uv(clamped_uv - vec2(0.0, axis.y))).rgb
    ) * 0.25;
    vec3 diagonal = (
        texture(planar_reflection, clamp_uv(clamped_uv + axis)).rgb +
        texture(planar_reflection, clamp_uv(clamped_uv + vec2(axis.x, -axis.y))).rgb +
        texture(planar_reflection, clamp_uv(clamped_uv + vec2(-axis.x, axis.y))).rgb +
        texture(planar_reflection, clamp_uv(clamped_uv - axis)).rgb
    ) * 0.25;

    return mix(center, mix(cross, diagonal, 0.35), blur);
}

vec4 planar_reflection_sample(vec3 world_pos, vec3 normal, float roughness) {
    if (scene.planar_texture_info.x < 0.5 || scene.planar_texture_info.w > 0.5) {
        return vec4(0.0);
    }

    vec3 plane_normal = normalize(scene.planar_plane.xyz);
    vec3 view_dir = normalize(world_pos - scene.camera_pos.xyz);
    vec3 reflected_view = reflect(view_dir, normal);
    float alignment = max(dot(normal, plane_normal), 0.0);
    float normal_weight = smoothstep(scene.planar_params.z, 1.0, alignment);
    float view_plane_alignment = abs(dot(-view_dir, plane_normal));
    float angle_weight = 1.0 - smoothstep(0.82, 0.98, view_plane_alignment);
    float roughness_limit = max(scene.planar_params.y, 0.04);
    float roughness_weight = 1.0 - smoothstep(roughness_limit * 0.7, roughness_limit, roughness);
    float plane_distance = abs(dot(world_pos, plane_normal) + scene.planar_plane.w);
    float distance_weight = 1.0 - smoothstep(0.0, max(scene.planar_params.w, 0.001), plane_distance);

    float plane_denom = dot(reflected_view, plane_normal);
    if (plane_denom >= -0.0005) {
        return vec4(0.0);
    }

    float plane_t = -(dot(world_pos, plane_normal) + scene.planar_plane.w) / plane_denom;
    if (plane_t <= 0.0) {
        return vec4(0.0);
    }

    vec3 reflected_hit = world_pos + reflected_view * plane_t;

    vec4 projected = scene.planar_view_proj * vec4(reflected_hit, 1.0);
    if (projected.w <= 0.0) {
        return vec4(0.0);
    }

    vec2 uv = projected.xy / projected.w * 0.5 + 0.5;
    if (scene.planar_texture_info.z > 0.5) {
        uv.y = 1.0 - uv.y;
    }

    vec2 edge_fade = vec2(0.04);
    vec2 uv_min = smoothstep(vec2(0.0), edge_fade, uv);
    vec2 uv_max = 1.0 - smoothstep(vec2(1.0) - edge_fade, vec2(1.0), uv);
    float edge_weight = uv_min.x * uv_min.y * uv_max.x * uv_max.y;
    float weight = scene.planar_params.x * normal_weight * angle_weight * roughness_weight * distance_weight * edge_weight;
    weight = smoothstep(0.0, 1.0, saturate(weight));

    return vec4(planar_reflection_color(uv, roughness), weight);
}

void main() {
    vec3 normal = normalize(frag_normal);
    if (object.texture_flags.z > 0.5) {
        normal = normal_from_map(normal, frag_world_pos, frag_uv);
    }

    vec3 view = normalize(scene.camera_pos.xyz - frag_world_pos);
    float metallic = clamp(object.material.x, 0.0, 1.0);
    float roughness = clamp(object.material.y, 0.04, 1.0);
    float specular_strength = clamp(object.material.z, 0.0, 1.0);
    float ao = clamp(object.material.w, 0.0, 1.0);
    vec3 emissive = object.emissive_color.rgb;
    vec4 base = frag_base_color;

    if (object.texture_flags.y > 0.5) {
        vec4 mr_texel = texture(metallic_roughness_texture, frag_uv);
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
    if (object.texture_flags.x > 0.5) {
        base *= texture(base_color_texture, frag_uv);
    }

    if (scene.planar_texture_info.w > 0.5) {
        float plane_side = dot(frag_world_pos, normalize(scene.planar_plane.xyz)) + scene.planar_plane.w;
        if (plane_side <= scene.planar_texture_info.y) {
            discard;
        }
    }

    bool sample_reflection_probe = scene.reflection_params.w > 0.5;
    if (!sample_reflection_probe) {
        metallic = 0.0;
        roughness = max(roughness, 0.65);
        specular_strength = min(specular_strength, 0.25);
    }

    float view_dot = max(dot(normal, view), 0.0);
    vec3 f0 = mix(vec3(0.04 * specular_strength), base.rgb, metallic);
    vec3 reflection = reflect(-view, normal);
    vec3 env = sample_reflection_probe ? environment_reflection(reflection, roughness, frag_world_pos) : vec3(0.0);
    vec3 env_fresnel = fresnel_schlick(view_dot, f0);
    vec3 diffuse_ambient = base.rgb * scene.ambient.rgb * ao * (1.0 - metallic);
    vec3 specular_ambient = env * env_fresnel * ao * mix(1.25, 0.35, roughness);
    vec4 planar = planar_reflection_sample(frag_world_pos, normal, roughness);
    vec3 planar_specular = planar.rgb * env_fresnel * ao * mix(1.35, 0.55, roughness);
    float planar_mix = planar.a * (1.0 - 0.35 * roughness);
    specular_ambient = mix(specular_ambient, planar_specular, saturate(planar_mix));
    vec3 directional = brdf_light(
        normal,
        view,
        normalize(-scene.light_dir.xyz),
        scene.light_color.rgb,
        base.rgb,
        metallic,
        roughness,
        specular_strength
    );

    if (base.a < object.texture_info.w) {
        discard;
    }

    vec3 color = diffuse_ambient + specular_ambient + directional;

    int point_count = int(scene.point_light_count.x + 0.5);
    for (int i = 0; i < MAX_EMISSIVE_LIGHTS; i++) {
        if (i >= point_count) {
            break;
        }

        vec4 position_radius = scene.point_light_pos_radius[i];
        vec4 color_power = scene.point_light_color_power[i];
        vec3 to_light = position_radius.xyz - frag_world_pos;
        float distance2 = max(dot(to_light, to_light), 0.0001);
        float radius = max(position_radius.w, 0.35);
        float attenuation = 1.0 / (1.0 + distance2 / (radius * radius * 5.0));
        vec3 point_light = to_light * inversesqrt(distance2);
        vec3 radiance = color_power.rgb * color_power.w * attenuation;

        color += brdf_light(
            normal,
            view,
            point_light,
            radiance,
            base.rgb,
            metallic,
            roughness,
            specular_strength
        );
    }

    color += emissive;

    out_color = vec4(color, base.a);
}
