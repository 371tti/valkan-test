#ifndef REBUILD1_PBR_LIGHTING_GLSL
#define REBUILD1_PBR_LIGHTING_GLSL

#include "common_math.glsl"

const float PBR_PI = 3.14159265359;
const float INV_PBR_PI = 0.31830988618;

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
    float ndoth = saturate(dot(normal, half_vector));
    float denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;

    // Keep the denominator floor tiny so low-roughness highlights stay sharp.
    return alpha2 / max(PBR_PI * denom * denom, 0.0000001);
}

float geometry_schlick_ggx(float ndot, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    return ndot / max(ndot * (1.0 - k) + k, 0.0001);
}

float geometry_smith(float ndotv, float ndotl, float roughness) {
    return geometry_schlick_ggx(ndotv, roughness) *
        geometry_schlick_ggx(ndotl, roughness);
}

float specular_occlusion(float ndotv, float ao, float roughness) {
    // A compact Frostbite/Filament-style approximation that avoids crushing metals.
    float exponent = exp2(-16.0 * roughness - 1.0);
    return clamp(pow(ndotv + ao, exponent) - 1.0 + ao, 0.0, 1.0);
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
    return mix(1.0, rcp_safe(max(alpha, 0.22), 0.22), transparency * 0.55);
}

float shadow_visibility(vec3 shadow) {
    return max(max(shadow.r, shadow.g), shadow.b);
}

vec3 direct_specular_shadow(vec3 shadow, float roughness, float visibility) {
    return shadow * mix(visibility, 1.0, roughness);
}

float reflected_directional_shadow(
    vec3 reflection,
    vec3 light,
    vec3 shadow,
    float roughness,
    float visibility
) {
    float smoothness = 1.0 - roughness;
    float alignment = saturate(dot(reflection, light));
    float directional_lobe = smoothstep(
        mix(0.70, 0.965, smoothness),
        1.0,
        alignment
    );
    float occlusion =
        (1.0 - visibility) *
        directional_lobe *
        mix(0.28, 1.0, smoothness);

    return 1.0 - occlusion;
}

vec3 pbr_direct_unshadowed(
    vec3 base_color,
    vec3 normal,
    vec3 view,
    vec3 light,
    vec3 radiance,
    float ndotv,
    vec3 f0,
    float metallic,
    float roughness,
    float spec_boost,
    float ndotl
) {
    if (ndotl <= 0.0001) {
        return vec3(0.0);
    }

    vec3 half_vector = normalize_fast(light + view);
    float vdoth = saturate(dot(view, half_vector));

    vec3 fresnel = fresnel_schlick(vdoth, f0);
    vec3 diffuse = (vec3(1.0) - fresnel) *
        ((1.0 - metallic) * INV_PBR_PI) *
        base_color;

    float distribution = distribution_ggx(normal, half_vector, roughness);
    float geometry = geometry_smith(ndotv, ndotl, roughness);
    vec3 specular = distribution *
        geometry *
        fresnel *
        rcp_safe(4.0 * ndotv * ndotl, 0.0001);
    specular *= spec_boost;

    return (diffuse + specular) * radiance * ndotl;
}

float local_light_visibility(
    vec3 normal,
    vec3 light,
    float light_distance,
    float light_radius
) {
    int caster_count = int(clamp(frame_camera.local_shadow_caster_count.x, 0.0, 8.0));
    if (caster_count <= 0 || light_distance <= 0.0001) {
        return 1.0;
    }

    vec3 ray_dir = light;
    vec3 light_position = frag_world_pos + ray_dir * light_distance;
    vec3 origin = frag_world_pos + normal * max(0.025, light_distance * 0.0008);
    float visibility = 1.0;

    for (int i = 0; i < 8; i++) {
        if (i >= caster_count) {
            break;
        }

        vec4 caster = frame_camera.local_shadow_caster_center_radius[i];
        float caster_radius = max(caster.w, 0.001);
        float caster_radius_sq = caster_radius * caster_radius;
        float influence_radius = (light_radius + caster_radius) * 1.15;
        vec3 light_to_caster = caster.xyz - light_position;
        if (dot(light_to_caster, light_to_caster) > influence_radius * influence_radius) {
            continue;
        }

        vec3 receiver_to_caster = caster.xyz - frag_world_pos;
        if (dot(receiver_to_caster, receiver_to_caster) <= caster_radius_sq * 1.08) {
            continue;
        }

        vec3 origin_to_caster = caster.xyz - origin;
        float ray_t = dot(origin_to_caster, ray_dir);
        if (ray_t <= 0.0 || ray_t >= light_distance) {
            continue;
        }

        float closest_sq = max(
            dot(origin_to_caster, origin_to_caster) - ray_t * ray_t,
            0.0
        );
        float soft_radius =
            caster_radius +
            light_radius * 0.10 +
            (light_distance - ray_t) * 0.08;
        float core_radius = max(caster_radius * 0.62, caster_radius - light_radius * 0.06);
        float occlusion = 1.0 - smoothstep(
            core_radius * core_radius,
            soft_radius * soft_radius,
            closest_sq
        );

        float segment = smoothstep(0.0, caster_radius * 0.65, ray_t) *
            smoothstep(0.0, caster_radius * 0.65, light_distance - ray_t);
        visibility = min(visibility, 1.0 - occlusion * segment * 0.82);
    }

    return clamp(visibility, 0.14, 1.0);
}

vec3 emissive_mesh_lighting(
    vec3 base_color,
    vec3 normal,
    vec3 view,
    float ndotv,
    vec3 f0,
    float metallic,
    float roughness,
    float spec_boost
) {
    vec3 direct = vec3(0.0);
    int light_count = int(clamp(frame_camera.emissive_light_count.x, 0.0, 4.0));
    if (light_count <= 0) {
        return direct;
    }

    for (int i = 0; i < 4; i++) {
        if (i >= light_count) {
            break;
        }

        vec4 light_position_radius = frame_camera.emissive_light_position_radius[i];
        float radius = max(light_position_radius.w, 0.001);
        float radius_sq = radius * radius;
        vec3 to_light = light_position_radius.xyz - frag_world_pos;
        float distance_sq = dot(to_light, to_light);

        if (distance_sq >= radius_sq || distance_sq <= 0.000001) {
            continue;
        }

        float falloff = 1.0 - distance_sq * rcp_safe(radius_sq, 0.0001);
        falloff *= falloff;
        falloff *= radius_sq * rcp_safe(distance_sq + radius_sq * 0.35, 0.0001);

        float inv_distance = inversesqrt(max(distance_sq, 0.000001));
        float light_distance = distance_sq * inv_distance;
        vec3 light = to_light * inv_distance;
        float ndotl = dot(normal, light);
        if (ndotl <= 0.0001) {
            continue;
        }
        float visibility = local_light_visibility(
            normal,
            light,
            light_distance,
            radius
        );
        vec3 radiance = frame_camera.emissive_light_color[i].rgb *
            falloff *
            visibility;
        direct += pbr_direct_unshadowed(
            base_color,
            normal,
            view,
            light,
            radiance,
            ndotv,
            f0,
            metallic,
            roughness,
            spec_boost,
            ndotl
        );
    }

    return direct;
}

vec4 pbr_shadow(float ndotl) {
    vec3 view_pos = (frame_camera.view * vec4(frag_world_pos, 1.0)).xyz;
    float camera_depth = max(-view_pos.z, 0.0);
    return shadow_factor(
        shadow_cascade_0,
        shadow_cascade_1,
        shadow_cascade_2,
        shadow_cascade_3,
        raw_shadow_cascade_0,
        raw_shadow_cascade_1,
        raw_shadow_cascade_2,
        raw_shadow_cascade_3,
        translucent_shadow_0,
        translucent_shadow_1,
        translucent_shadow_2,
        translucent_shadow_3,
        frag_shadow_pos,
        camera_depth,
        frame_camera.shadow_cascade_splits,
        frame_camera.shadow_cascade_texel_world,
        frame_camera.shadow_cascade_depth_span,
        frame_camera.contact_shadow,
        frame_camera.light_color.w,
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
    base_color = max(base_color, vec3(0.0));
    roughness = clamp(roughness, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    occlusion = clamp(occlusion, 0.0, 1.0);
    vec3 view = normalize_fast(frame_camera.camera_pos.xyz - frag_world_pos);
    float ndotv = max(dot(normal, view), 0.0001);

    vec3 f0 = mix(vec3(0.04), base_color, metallic);
    float spec_boost = mix(transparent_specular_boost(alpha), 1.0, metallic);
    float smoothness = 1.0 - roughness;

    vec3 direct = vec3(0.0);
    vec3 light = vec3(0.0, 1.0, 0.0);
    vec4 shadow = vec4(1.0);
    float directional_visibility = 1.0;
    bool has_directional_light = frame_camera.light_dir.w > 0.5;

    if (has_directional_light) {
        light = -frame_camera.light_dir.xyz;
        float ndotl = saturate(dot(normal, light));

        if (ndotl > 0.0001) {
            shadow = pbr_shadow(ndotl);
            directional_visibility = shadow_visibility(shadow.rgb);

            vec3 half_vector = normalize_fast(light + view);
            float vdoth = saturate(dot(view, half_vector));

            vec3 fresnel = fresnel_schlick(vdoth, f0);
            vec3 diffuse = (vec3(1.0) - fresnel) *
                ((1.0 - metallic) * INV_PBR_PI) *
                base_color;

            float distribution = distribution_ggx(normal, half_vector, roughness);
            float geometry = geometry_smith(ndotv, ndotl, roughness);
            vec3 specular = distribution *
                geometry *
                fresnel *
                rcp_safe(4.0 * ndotv * ndotl, 0.0001);
            specular *= spec_boost;

            float diffuse_contact = clamp(shadow.a, 0.0, 1.0);
            float contact_occlusion = 1.0 - diffuse_contact;
            float specular_contact_strength =
                mix(0.62, 1.0, roughness) *
                (1.0 - metallic * smoothness * 0.32);
            float specular_contact =
                1.0 - contact_occlusion * specular_contact_strength;
            vec3 specular_shadow =
                direct_specular_shadow(
                    shadow.rgb,
                    roughness,
                    directional_visibility
                );

            direct = (
                    diffuse * diffuse_contact * shadow.rgb +
                    specular * specular_contact * specular_shadow
                ) *
                frame_camera.light_color.rgb *
                ndotl;
        }
    }

    direct += emissive_mesh_lighting(
        base_color,
        normal,
        view,
        ndotv,
        f0,
        metallic,
        roughness,
        spec_boost
    );

    float indirect_strength = max(frame_camera.ambient_color.w, 0.0);

    vec3 diffuse_irradiance =
        frame_camera.ambient_color.rgb +
        hemisphere_light(normal) * indirect_strength;

    vec3 indirect_diffuse =
        base_color *
        (1.0 - metallic) *
        diffuse_irradiance *
        occlusion;

    vec3 reflection = reflect(-view, normal);
    vec3 specular_env =
        frame_camera.ambient_color.rgb * 0.35 +
        hemisphere_light(reflection) * indirect_strength;

    vec3 indirect_fresnel = fresnel_schlick_roughness(ndotv, f0, roughness);
    float roughness_energy = 0.18 + smoothness * smoothness * 0.82;
    float sao = specular_occlusion(ndotv, occlusion, roughness);
    float reflected_light_shadow = has_directional_light
        ? reflected_directional_shadow(
            reflection,
            light,
            shadow.rgb,
            roughness,
            directional_visibility
        )
        : 1.0;

    vec3 indirect_specular =
        specular_env *
        indirect_fresnel *
        roughness_energy *
        sao *
        spec_boost *
        reflected_light_shadow;

    return indirect_diffuse + indirect_specular + direct + emissive;
}

float material_reflectance(vec3 base_color, float metallic, float roughness) {
    metallic = clamp(metallic, 0.0, 1.0);
    roughness = clamp(roughness, 0.04, 1.0);

    // This is a visibility weight for post SSR, not strict physical F0.
    float conductor = max(max(base_color.r, base_color.g), base_color.b);
    float dielectric = mix(0.14, 0.055, roughness);
    return clamp(mix(dielectric, conductor, metallic), 0.0, 1.0);
}

vec2 oct_encode(vec3 normal) {
    normal = normalize_fast(normal);
    normal *= rcp_safe(abs(normal.x) + abs(normal.y) + abs(normal.z), 0.0001);
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
