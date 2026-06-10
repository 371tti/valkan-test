#ifndef REBUILD1_PBR_LIGHTING_GLSL
#define REBUILD1_PBR_LIGHTING_GLSL

const float PBR_PI = 3.14159265359;
const float INV_PBR_PI = 0.31830988618;

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

float pow5(float value) {
    float value2 = value * value;
    return value2 * value2 * value;
}

float pbr_rcp_safe(float value, float floor_value) {
    return 1.0 / max(value, floor_value);
}

vec3 pbr_normalize_fast(vec3 value) {
    return value * inversesqrt(max(dot(value, value), 0.000001));
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow5(saturate(1.0 - cos_theta));
}

vec3 fresnel_schlick_roughness(float cos_theta, vec3 f0, float roughness) {
    vec3 rough_f0 = max(vec3(1.0 - roughness), f0);
    return f0 + (rough_f0 - f0) * pow5(saturate(1.0 - cos_theta));
}

float distribution_ggx(vec3 normal, vec3 half_vector, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float ndoth = max(dot(normal, half_vector), 0.0);
    float denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;
    return alpha2 * pbr_rcp_safe(PBR_PI * denom * denom, 0.0001);
}

float geometry_schlick_ggx(float ndot, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    return ndot * pbr_rcp_safe(ndot * (1.0 - k) + k, 0.0001);
}

float geometry_smith(float ndotv, float ndotl, float roughness) {
    return geometry_schlick_ggx(ndotv, roughness) *
        geometry_schlick_ggx(ndotl, roughness);
}

vec3 hemisphere_light(vec3 direction) {
    float sky = saturate(direction.y * 0.5 + 0.5);
    vec3 ground_color = vec3(0.030, 0.025, 0.020);
    vec3 horizon_color = vec3(0.060, 0.065, 0.076);
    vec3 sky_color = vec3(0.155, 0.185, 0.240);
    vec3 lower = mix(ground_color, horizon_color, sky);
    vec3 upper = mix(horizon_color, sky_color, sky * sky);
    return mix(lower, upper, sky);
}

float transparent_specular_boost(float alpha) {
    float transparency = 1.0 - clamp(alpha, 0.0, 1.0);
    return mix(1.0, pbr_rcp_safe(max(alpha, 0.22), 0.22), transparency * 0.55);
}

vec3 pbr_shadow(float ndotl) {
    if (frame_camera.light_dir.w <= 0.5 || ndotl <= 0.0001) {
        return vec3(1.0);
    }

    vec3 camera_delta = frame_camera.camera_pos.xyz - frag_world_pos;
    float camera_distance_sq = dot(camera_delta, camera_delta);
    return shadow_factor(
        shadow_cascade_0,
        shadow_cascade_1,
        shadow_cascade_2,
        translucent_shadow_0,
        translucent_shadow_1,
        translucent_shadow_2,
        frag_shadow_pos,
        camera_distance_sq,
        frame_camera.shadow_cascade_splits,
        ndotl
    );
}

vec3 shade_pbr(
    vec3 base_color,
    vec3 normal,
    float metallic,
    float roughness,
    float occlusion,
    vec3 emissive,
    float alpha
) {
    roughness = clamp(roughness, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    occlusion = clamp(occlusion, 0.0, 1.0);

    vec3 view = pbr_normalize_fast(frame_camera.camera_pos.xyz - frag_world_pos);
    vec3 light = -frame_camera.light_dir.xyz;
    vec3 half_vector = pbr_normalize_fast(light + view);
    float ndotv = max(dot(normal, view), 0.0001);
    float ndotl = max(dot(normal, light), 0.0);
    float vdoth = max(dot(view, half_vector), 0.0);
    vec3 f0 = mix(vec3(0.04), base_color, metallic);
    vec3 fresnel = fresnel_schlick(vdoth, f0);
    vec3 diffuse = (vec3(1.0) - fresnel) * ((1.0 - metallic) * INV_PBR_PI) * base_color;
    float distribution = distribution_ggx(normal, half_vector, roughness);
    float geometry = geometry_smith(ndotv, ndotl, roughness);
    vec3 specular = distribution * geometry * fresnel *
        pbr_rcp_safe(4.0 * ndotv * max(ndotl, 0.0001), 0.0001);
    specular *= transparent_specular_boost(alpha);

    vec3 shadow = pbr_shadow(ndotl);
    vec3 direct = (diffuse + specular) * frame_camera.light_color.rgb * ndotl * shadow;

    float indirect_strength = max(frame_camera.ambient_color.w, 0.0);
    vec3 irradiance = frame_camera.ambient_color.rgb + hemisphere_light(normal) * indirect_strength;
    vec3 indirect_diffuse = base_color * (1.0 - metallic) * irradiance * occlusion;
    vec3 reflection = reflect(-view, normal);
    vec3 specular_ibl_color = hemisphere_light(reflection) * indirect_strength;
    vec3 indirect_fresnel = fresnel_schlick_roughness(ndotv, f0, roughness);
    float smoothness = 1.0 - roughness;
    vec3 indirect_specular = specular_ibl_color * indirect_fresnel *
        (0.18 + smoothness * smoothness * 0.82) * occlusion * transparent_specular_boost(alpha);

    return indirect_diffuse + indirect_specular + direct + emissive;
}

float material_reflectance(vec3 base_color, float metallic, float roughness) {
    float conductor = max(max(base_color.r, base_color.g), base_color.b);
    float dielectric = 0.14 - 0.105 * roughness;
    return clamp(mix(dielectric, conductor, metallic), 0.0, 1.0);
}

vec2 sign_not_zero(vec2 value) {
    return vec2(value.x >= 0.0 ? 1.0 : -1.0, value.y >= 0.0 ? 1.0 : -1.0);
}

vec2 oct_encode(vec3 normal) {
    normal *= pbr_rcp_safe(abs(normal.x) + abs(normal.y) + abs(normal.z), 0.0001);
    if (normal.z < 0.0) {
        normal.xy = (1.0 - abs(normal.yx)) * sign_not_zero(normal.xy);
    }
    return normal.xy * 0.5 + 0.5;
}

vec4 pack_view_normal_material(vec3 normal, float roughness, float metallic, vec3 base_color) {
    vec3 view_normal = (frame_camera.view * vec4(normal, 0.0)).xyz;
    return vec4(
        oct_encode(view_normal),
        clamp(roughness, 0.04, 1.0),
        material_reflectance(base_color, metallic, roughness)
    );
}

#endif
