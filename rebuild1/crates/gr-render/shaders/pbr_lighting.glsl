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
    if (ao >= 1.0) {
        return 1.0;
    }

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

float local_light_falloff_in_range(float distance_sq, float range_sq) {
    float falloff = 1.0 - distance_sq * rcp_safe(range_sq, 0.0001);
    falloff *= falloff;
    return falloff * range_sq * rcp_safe(distance_sq + range_sq * 0.35, 0.0001);
}

bool local_light_vector(
    vec3 sample_position,
    float range_sq,
    out vec3 light,
    out float distance_sq,
    out float light_distance
) {
    vec3 to_light = sample_position - frag_world_pos;
    distance_sq = dot(to_light, to_light);
    if (distance_sq >= range_sq || distance_sq <= 0.000001) {
        return false;
    }

    float inv_distance = inversesqrt(distance_sq);
    light_distance = distance_sq * inv_distance;
    light = to_light * inv_distance;
    return true;
}

float area_adjusted_roughness(float roughness, float source_radius, float light_distance) {
    if (source_radius <= 0.0001) {
        return roughness;
    }

    float angular_size = source_radius * rcp_safe(light_distance, 0.0001);
    return clamp(sqrt(roughness * roughness + angular_size * angular_size), roughness, 1.0);
}

void local_shadow_basis(vec3 direction, out vec3 tangent, out vec3 bitangent) {
    vec3 helper = abs(direction.y) < 0.92 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    tangent = normalize_fast(cross(helper, direction));
    bitangent = cross(direction, tangent);
}

float local_shadow_depth_from_distance(float distance, float near_plane, float far_plane) {
    float safe_distance = max(distance, near_plane + 0.0001);
    return (far_plane * (safe_distance - near_plane))
        * rcp_safe(safe_distance * (far_plane - near_plane), 0.0001);
}

float local_shadow_cube_face_distance(vec3 direction, float light_distance) {
    // The depth cubemap stores perspective face depth, not radial point-light distance.
    vec3 axis = abs(direction);
    float major_axis = max(max(axis.x, axis.y), axis.z);
    return light_distance * major_axis;
}

float local_shadow_sample_depth(int shadow_index, vec3 direction) {
    if (shadow_index == 0) {
        return texture(local_shadow_depth[0], direction).r;
    }
    if (shadow_index == 1) {
        return texture(local_shadow_depth[1], direction).r;
    }
    if (shadow_index == 2) {
        return texture(local_shadow_depth[2], direction).r;
    }
    return texture(local_shadow_depth[3], direction).r;
}

float local_shadow_compare(
    int shadow_index,
    vec3 direction,
    float compare_depth,
    float bias,
    float angular_radius
) {
    vec3 tangent;
    vec3 bitangent;
    local_shadow_basis(direction, tangent, bitangent);

    float shadow_depth = compare_depth - bias;
    float visibility = 0.0;
    visibility += step(shadow_depth, local_shadow_sample_depth(shadow_index, direction)) * 0.40;
    visibility += step(
        shadow_depth,
        local_shadow_sample_depth(shadow_index, normalize_fast(direction + tangent * angular_radius))
    ) * 0.15;
    visibility += step(
        shadow_depth,
        local_shadow_sample_depth(shadow_index, normalize_fast(direction - tangent * angular_radius))
    ) * 0.15;
    visibility += step(
        shadow_depth,
        local_shadow_sample_depth(shadow_index, normalize_fast(direction + bitangent * angular_radius))
    ) * 0.15;
    visibility += step(
        shadow_depth,
        local_shadow_sample_depth(shadow_index, normalize_fast(direction - bitangent * angular_radius))
    ) * 0.15;
    return visibility;
}

float local_light_visibility(
    int light_index,
    vec3 normal,
    vec3 light,
    float light_distance,
    float ndotl,
    float source_radius,
    float shadow_strength
) {
    if (shadow_strength <= 0.0) {
        return 1.0;
    }

    int shadow_index = clamp(light_index, 0, 3);
    vec4 shadow_params = frame_camera.local_shadow_params[shadow_index];
    if (shadow_params.x <= 0.5 || light_distance <= 0.0001) {
        return 1.0;
    }

    vec3 light_to_receiver = -light;
    float cube_face_distance = local_shadow_cube_face_distance(light_to_receiver, light_distance);
    float near_plane = max(shadow_params.z, 0.001);
    float far_plane = max(shadow_params.w, near_plane + 0.1);
    if (cube_face_distance <= near_plane || cube_face_distance >= far_plane) {
        return 1.0;
    }

    float receiver_depth = local_shadow_depth_from_distance(
        cube_face_distance,
        near_plane,
        far_plane
    );
    float slope_bias = 0.0012 + 0.0045 * (1.0 - clamp(ndotl, 0.0, 1.0));
    float source_bias = clamp(source_radius * rcp_safe(far_plane, 0.0001) * 0.015, 0.0, 0.006);
    float bias = slope_bias + source_bias;
    float texel_angle = shadow_params.y;
    float source_angle = max(source_radius, 0.001) * rcp_safe(light_distance, 0.0001);
    float angular_radius = max(texel_angle, source_angle * 0.08);
    float visibility = local_shadow_compare(
        shadow_index,
        light_to_receiver,
        receiver_depth,
        bias,
        angular_radius
    );

    return mix(1.0, clamp(visibility, 0.04, 1.0), clamp(shadow_strength * 1.12, 0.0, 1.0));
}

void local_rect_basis(vec3 direction, out vec3 right, out vec3 up) {
    vec3 helper = abs(direction.y) < 0.92 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    right = normalize_fast(cross(helper, direction));
    up = cross(direction, right);
}

vec3 local_light_resolved_lighting(
    vec3 base_color,
    vec3 normal,
    vec3 view,
    float ndotv,
    vec3 f0,
    float metallic,
    float roughness,
    float spec_boost,
    vec3 light,
    float distance_sq,
    float light_distance,
    float ndotl,
    vec3 radiance_color,
    float range_sq,
    float source_radius,
    float shadow_strength,
    float emission_facing,
    float sample_weight,
    int light_index
) {
    if (emission_facing <= 0.0001 || sample_weight <= 0.0001) {
        return vec3(0.0);
    }

    if (ndotl <= 0.0001) {
        return vec3(0.0);
    }

    float falloff = local_light_falloff_in_range(distance_sq, range_sq);
    if (falloff <= 0.000001) {
        return vec3(0.0);
    }

    float visibility = local_light_visibility(
        light_index,
        normal,
        light,
        light_distance,
        ndotl,
        source_radius,
        shadow_strength
    );
    float local_roughness = area_adjusted_roughness(
        roughness,
        source_radius,
        light_distance
    );
    vec3 radiance = radiance_color *
        falloff *
        visibility *
        emission_facing *
        sample_weight;

    return pbr_direct_unshadowed(
        base_color,
        normal,
        view,
        light,
        radiance,
        ndotv,
        f0,
        metallic,
        local_roughness,
        spec_boost,
        ndotl
    );
}

vec3 local_light_sample_lighting(
    vec3 base_color,
    vec3 normal,
    vec3 view,
    float ndotv,
    vec3 f0,
    float metallic,
    float roughness,
    float spec_boost,
    vec3 sample_position,
    vec3 radiance_color,
    float range_sq,
    float source_radius,
    float shadow_strength,
    float emission_facing,
    float sample_weight,
    int light_index
) {
    vec3 light;
    float distance_sq;
    float light_distance;
    if (!local_light_vector(sample_position, range_sq, light, distance_sq, light_distance)) {
        return vec3(0.0);
    }

    return local_light_resolved_lighting(
        base_color,
        normal,
        view,
        ndotv,
        f0,
        metallic,
        roughness,
        spec_boost,
        light,
        distance_sq,
        light_distance,
        dot(normal, light),
        radiance_color,
        range_sq,
        source_radius,
        shadow_strength,
        emission_facing,
        sample_weight,
        light_index
    );
}

float spot_light_facing(vec3 light_to_receiver, vec3 direction, vec2 cone_cos) {
    float inner_cos = max(cone_cos.x, cone_cos.y + 0.0001);
    float outer_cos = min(cone_cos.y, inner_cos - 0.0001);
    float angle_cos = dot(direction, light_to_receiver);
    float cone = smoothstep(outer_cos, inner_cos, angle_cos);
    return cone * cone;
}

vec3 spot_light_lighting(
    vec3 base_color,
    vec3 normal,
    vec3 view,
    float ndotv,
    vec3 f0,
    float metallic,
    float roughness,
    float spec_boost,
    vec3 position,
    vec3 radiance_color,
    float range_sq,
    vec3 direction,
    vec2 cone_cos,
    float source_radius,
    float shadow_strength,
    int light_index
) {
    vec3 light;
    float distance_sq;
    float light_distance;
    if (!local_light_vector(position, range_sq, light, distance_sq, light_distance)) {
        return vec3(0.0);
    }

    float facing = spot_light_facing(-light, direction, cone_cos);
    return local_light_resolved_lighting(
        base_color,
        normal,
        view,
        ndotv,
        f0,
        metallic,
        roughness,
        spec_boost,
        light,
        distance_sq,
        light_distance,
        dot(normal, light),
        radiance_color,
        range_sq,
        source_radius,
        shadow_strength,
        facing,
        1.0,
        light_index
    );
}

vec3 rectangular_area_light_lighting(
    vec3 base_color,
    vec3 normal,
    vec3 view,
    float ndotv,
    vec3 f0,
    float metallic,
    float roughness,
    float spec_boost,
    vec3 center,
    vec3 radiance_color,
    float range_sq,
    float reach_sq,
    vec3 direction,
    vec2 half_size,
    float source_radius,
    float shadow_strength,
    int light_index
) {
    vec3 center_to_receiver = center - frag_world_pos;
    if (dot(center_to_receiver, center_to_receiver) >= reach_sq) {
        return vec3(0.0);
    }

    vec3 right;
    vec3 up;
    local_rect_basis(direction, right, up);

    vec3 direct = vec3(0.0);
    vec2 offsets[5] = vec2[5](
        vec2(0.0, 0.0),
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0)
    );
    float weights[5] = float[5](0.36, 0.16, 0.16, 0.16, 0.16);

    for (int sample_index = 0; sample_index < 5; sample_index++) {
        vec2 offset = offsets[sample_index] * half_size;
        vec3 sample_position = center + right * offset.x + up * offset.y;
        vec3 light;
        float distance_sq;
        float light_distance;
        if (!local_light_vector(sample_position, range_sq, light, distance_sq, light_distance)) {
            continue;
        }

        float facing = saturate(dot(direction, -light));
        if (facing <= 0.0001) {
            continue;
        }

        direct += local_light_resolved_lighting(
            base_color,
            normal,
            view,
            ndotv,
            f0,
            metallic,
            roughness,
            spec_boost,
            light,
            distance_sq,
            light_distance,
            dot(normal, light),
            radiance_color,
            range_sq,
            source_radius,
            shadow_strength,
            facing,
            weights[sample_index],
            light_index
        );
    }

    return direct;
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
        vec4 light_direction_radius = frame_camera.emissive_light_direction_radius[i];
        vec4 light_size_kind = frame_camera.emissive_light_size_kind[i];
        vec4 light_color = frame_camera.emissive_light_color[i];
        if (light_color.a <= 0.0001) {
            continue;
        }

        vec3 radiance_color = light_color.rgb;
        float range = max(light_position_radius.w, 0.001);
        float range_sq = range * range;
        float source_radius = max(light_direction_radius.w, 0.0);
        float kind = light_size_kind.z;
        float shadow_strength = light_size_kind.w;

        if (kind > 2.5) {
            vec2 half_size = max(light_size_kind.xy, vec2(0.001));
            float rect_source_radius = max(source_radius, length(half_size));
            float reach = range + rect_source_radius;
            direct += rectangular_area_light_lighting(
                base_color,
                normal,
                view,
                ndotv,
                f0,
                metallic,
                roughness,
                spec_boost,
                light_position_radius.xyz,
                radiance_color,
                range_sq,
                reach * reach,
                light_direction_radius.xyz,
                half_size,
                rect_source_radius,
                shadow_strength,
                i
            );
        } else if (kind > 1.5) {
            direct += spot_light_lighting(
                base_color,
                normal,
                view,
                ndotv,
                f0,
                metallic,
                roughness,
                spec_boost,
                light_position_radius.xyz,
                radiance_color,
                range_sq,
                light_direction_radius.xyz,
                light_size_kind.xy,
                source_radius,
                shadow_strength,
                i
            );
        } else {
            direct += local_light_sample_lighting(
                base_color,
                normal,
                view,
                ndotv,
                f0,
                metallic,
                roughness,
                spec_boost,
                light_position_radius.xyz,
                radiance_color,
                range_sq,
                source_radius,
                shadow_strength,
                1.0,
                1.0,
                i
            );
        }
    }

    return direct;
}

vec4 pbr_shadow(float ndotl) {
    float camera_depth = max(frag_view_depth, 0.0);
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
    float mirror_surface =
        smoothstep(0.92, 0.995, metallic) *
        (1.0 - smoothstep(0.045, 0.085, roughness));
    float direct_lighting_weight = 1.0 - mirror_surface;
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

            if (direct_lighting_weight > 0.0001) {
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
    }

    if (direct_lighting_weight > 0.0001) {
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
        direct *= direct_lighting_weight;
    }

    float indirect_strength = max(frame_camera.ambient_color.w, 0.0);

    vec3 diffuse_irradiance =
        frame_camera.ambient_color.rgb +
        hemisphere_light(normal) * indirect_strength;

    vec3 indirect_diffuse =
        base_color *
        (1.0 - metallic) *
        diffuse_irradiance *
        occlusion *
        direct_lighting_weight;

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
    // This is a visibility weight for post SSR, not strict physical F0.
    float conductor = max(max(base_color.r, base_color.g), base_color.b);
    float dielectric = mix(0.14, 0.055, roughness);
    return clamp(mix(dielectric, conductor, metallic), 0.0, 1.0);
}

vec2 oct_encode_unit(vec3 normal) {
    normal *= rcp_safe(abs(normal.x) + abs(normal.y) + abs(normal.z), 0.0001);
    if (normal.z < 0.0) {
        normal.xy = (1.0 - abs(normal.yx)) * sign_not_zero(normal.xy);
    }
    return normal.xy * 0.5 + 0.5;
}

vec4 pack_view_normal_material(vec3 normal, float roughness, float metallic, vec3 base_color) {
    vec3 view_normal = (frame_camera.view * vec4(normal, 0.0)).xyz;
    return vec4(
        oct_encode_unit(view_normal),
        roughness,
        material_reflectance(base_color, metallic, roughness)
    );
}

#endif
