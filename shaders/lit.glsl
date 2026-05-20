#ifndef LIT_GLSL
#define LIT_GLSL

struct MaterialSample {
    vec4 base;
    vec3 normal;
    vec3 view;
    vec3 emissive;
    float metallic;
    float roughness;
    float specular;
    vec3 specular_color;
    float ao;
};

struct GiSample {
    vec3 diffuse;
    vec3 specular;
};

vec3 aces_tonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

vec3 apply_camera_response(vec3 color) {
    if (scene.camera_response.w < 0.5) {
        return color;
    }

    color = max(color, vec3(0.0)) * scene.camera_response.x * scene.white_balance.rgb;
    color = aces_tonemap(color);
    color = mix(vec3(luma(color)), color, scene.camera_response.z);
    color = mix(vec3(0.5), color, scene.camera_response.y);

    return clamp(color, 0.0, 1.0);
}

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

MaterialSample read_material(bool use_textures) {
    MaterialSample material;
    vec3 surface_normal = normalize(frag_normal);
    if (object.material_ext.w > 0.5 && !gl_FrontFacing) {
        surface_normal = -surface_normal;
    }
    material.normal = surface_normal;
    if (use_textures && object.texture_flags.z > 0.5) {
        material.normal = normal_from_map(surface_normal, frag_world_pos, frag_uv);
    }

    material.view = normalize(scene.camera_pos.xyz - frag_world_pos);
    material.metallic = clamp(object.material.x, 0.0, 1.0);
    material.roughness = clamp(object.material.y, 0.04, 1.0);
    material.specular = clamp(object.material.z, 0.0, 1.0);
    material.specular_color = clamp(object.material_ext.rgb, 0.0, 1.0);
    material.ao = clamp(object.material.w, 0.0, 1.0);
    material.emissive = object.emissive_color.rgb;
    material.base = frag_base_color;

    if (use_textures && object.texture_flags.y > 0.5) {
        vec4 mr_texel = texture(metallic_roughness_texture, frag_uv);
        material.roughness *= mr_texel.g;
        material.metallic *= mr_texel.b;
    }
    if (use_textures && object.texture_flags.w > 0.5) {
        float occlusion = texture(occlusion_texture, frag_uv).r;
        material.ao *= mix(1.0, occlusion, object.texture_info.z);
    }
    if (use_textures && object.texture_info.x > 0.5) {
        material.emissive *= texture(emissive_texture, frag_uv).rgb;
    }
    if (use_textures && object.texture_flags.x > 0.5) {
        material.base *= texture(base_color_texture, frag_uv);
    }

    return material;
}

void simplify_material_without_probe(inout MaterialSample material) {
    material.metallic *= 0.45;
    material.roughness = max(material.roughness, 0.5);
    material.specular = min(material.specular, 0.45);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

vec3 fresnel_schlick_roughness(float cos_theta, vec3 f0, float roughness) {
    vec3 grazing = max(vec3(1.0 - roughness), f0);

    return f0 + (grazing - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

float distribution_ggx(vec3 normal, vec3 half_vec, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float ndoth = max(dot(normal, half_vec), 0.0);
    float denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;

    return alpha2 / max(PI * denom * denom, 0.0001);
}

float geometry_schlick_ggx(float ndot, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;

    return ndot / max(ndot * (1.0 - k) + k, 0.0001);
}

float geometry_smith(vec3 normal, vec3 view, vec3 light, float roughness) {
    return geometry_schlick_ggx(max(dot(normal, view), 0.0), roughness)
        * geometry_schlick_ggx(max(dot(normal, light), 0.0), roughness);
}

vec3 material_f0(MaterialSample material) {
    float dielectric_f0 = mix(0.02, 0.16, material.specular);
    return mix(dielectric_f0 * material.specular_color, material.base.rgb, material.metallic);
}

vec3 material_direct_light(MaterialSample material, vec3 light, vec3 radiance) {
    vec3 normal = normalize(material.normal);
    vec3 view = normalize(material.view);
    vec3 half_vec = normalize(view + light);
    float ndotv = max(dot(normal, view), 0.0);
    float ndotl = max(dot(normal, light), 0.0);
    float vdoth = max(dot(view, half_vec), 0.0);
    if (ndotv <= 0.0001 || ndotl <= 0.0001) {
        return vec3(0.0);
    }

    vec3 f0 = material_f0(material);
    vec3 fresnel = fresnel_schlick(vdoth, f0);
    float distribution = distribution_ggx(normal, half_vec, material.roughness);
    float geometry = geometry_smith(normal, view, light, material.roughness);
    vec3 specular = distribution * geometry * fresnel / max(4.0 * ndotv * ndotl, 0.0001);
    vec3 diffuse = (vec3(1.0) - fresnel)
        * (1.0 - material.metallic)
        * material.base.rgb
        / PI;

    return (diffuse + specular) * radiance * ndotl;
}

vec3 parallax_correct_reflection(vec3 world_pos, vec3 reflection) {
    if (scene.reflection_probe_pos_radius.w <= 0.0 || scene.reflection_params.y < 0.5) {
        return reflection;
    }

    vec3 direction = safe_direction(normalize(reflection));
    vec3 box_min = scene.reflection_probe_box_min.xyz;
    vec3 box_max = scene.reflection_probe_box_max.xyz;
    vec3 far_plane = max((box_min - world_pos) / direction, (box_max - world_pos) / direction);
    float distance = min(min(far_plane.x, far_plane.y), far_plane.z);

    if (distance <= 0.0) {
        return reflection;
    }

    vec3 hit = world_pos + direction * distance;
    vec3 center_to_world = world_pos - scene.reflection_probe_pos_radius.xyz;
    float probe_radius = max(scene.reflection_probe_pos_radius.w, 0.001);
    float parallax_weight = smoothstep(0.05, 0.85, clamp(length(center_to_world) / probe_radius, 0.0, 1.0));

    return normalize(mix(reflection, normalize(hit - scene.reflection_probe_pos_radius.xyz), parallax_weight));
}

vec3 reflection_probe_lodless(vec3 direction, float roughness) {
    direction = normalize(direction);
    if (roughness < 0.12) {
        return texture(reflection_probe, direction).rgb;
    }

    vec3 up = abs(direction.y) > 0.88 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 tangent = normalize(cross(up, direction));
    vec3 bitangent = cross(direction, tangent);
    float spread = roughness * roughness * 0.72;
    vec3 color = texture(reflection_probe, direction).rgb * 4.0;
    float weight_sum = 4.0;
    const vec2 taps[8] = vec2[](
        vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0),
        vec2(0.707, 0.707), vec2(-0.707, 0.707), vec2(0.707, -0.707), vec2(-0.707, -0.707)
    );

    for (int i = 0; i < 8; i++) {
        float ring = i < 4 ? 0.68 : 1.15;
        float weight = i < 4 ? 1.15 : 0.85;
        vec3 sample_dir = normalize(direction + (tangent * taps[i].x + bitangent * taps[i].y) * spread * ring);
        color += texture(reflection_probe, sample_dir).rgb * weight;
        weight_sum += weight;
    }

    return color / weight_sum;
}

vec3 environment_reflection(vec3 reflection, float roughness, vec3 world_pos) {
    if (scene.camera_pos.w > 0.5 && scene.reflection_params.w > 0.5) {
        vec3 probe_direction = parallax_correct_reflection(world_pos, reflection);
        vec3 probe = reflection_probe_lodless(probe_direction, roughness);
        vec3 fallback = environment_color(reflection);
        float probe_radius = max(scene.reflection_probe_pos_radius.w, 0.001);
        float world_to_center = length(world_pos - scene.reflection_probe_pos_radius.xyz);
        float probe_coverage = 1.0 - smoothstep(probe_radius * 0.65, probe_radius * 1.35, world_to_center);
        vec3 reflected = mix(probe, fallback, roughness * roughness * scene.reflection_params.z);

        return mix(fallback, reflected, scene.reflection_params.x * probe_coverage);
    }

    return environment_color(reflection);
}

vec3 planar_reflection_color(vec2 uv, float roughness) {
    vec2 clamped_uv = clamp(uv, vec2(0.0), vec2(1.0));
    vec2 texel = 1.0 / vec2(textureSize(planar_reflection, 0));
    float blur = roughness * roughness;
    vec2 axis = texel * mix(1.0, 6.0, blur);

    vec3 center = texture(planar_reflection, clamped_uv).rgb;
    vec3 cross = (
        texture(planar_reflection, clamp(clamped_uv + vec2(axis.x, 0.0), vec2(0.0), vec2(1.0))).rgb +
        texture(planar_reflection, clamp(clamped_uv - vec2(axis.x, 0.0), vec2(0.0), vec2(1.0))).rgb +
        texture(planar_reflection, clamp(clamped_uv + vec2(0.0, axis.y), vec2(0.0), vec2(1.0))).rgb +
        texture(planar_reflection, clamp(clamped_uv - vec2(0.0, axis.y), vec2(0.0), vec2(1.0))).rgb
    ) * 0.25;
    vec3 diagonal = (
        texture(planar_reflection, clamp(clamped_uv + axis, vec2(0.0), vec2(1.0))).rgb +
        texture(planar_reflection, clamp(clamped_uv + vec2(axis.x, -axis.y), vec2(0.0), vec2(1.0))).rgb +
        texture(planar_reflection, clamp(clamped_uv + vec2(-axis.x, axis.y), vec2(0.0), vec2(1.0))).rgb +
        texture(planar_reflection, clamp(clamped_uv - axis, vec2(0.0), vec2(1.0))).rgb
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
    float angle_weight = 1.0 - smoothstep(0.82, 0.98, abs(dot(-view_dir, plane_normal)));
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

    vec4 projected = scene.planar_view_proj * vec4(world_pos + reflected_view * plane_t, 1.0);
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
    float weight = scene.planar_params.x * normal_weight * angle_weight * roughness_weight
        * distance_weight * uv_min.x * uv_min.y * uv_max.x * uv_max.y;

    return vec4(planar_reflection_color(uv, roughness), smoothstep(0.0, 1.0, saturate(weight)));
}

float sh0(vec3 n) { return 0.282095; }
float sh1(vec3 n) { return 0.488603 * n.y; }
float sh2(vec3 n) { return 0.488603 * n.z; }
float sh3(vec3 n) { return 0.488603 * n.x; }
float sh4(vec3 n) { return 1.092548 * n.x * n.y; }
float sh5(vec3 n) { return 1.092548 * n.y * n.z; }
float sh6(vec3 n) { return 0.315392 * (3.0 * n.z * n.z - 1.0); }
float sh7(vec3 n) { return 1.092548 * n.x * n.z; }
float sh8(vec3 n) { return 0.546274 * (n.x * n.x - n.y * n.y); }

vec3 irradiance_probe(vec3 normal) {
    vec3 n = normalize(normal);
    vec3 irradiance =
        scene.gi_sh[0].rgb * sh0(n) +
        scene.gi_sh[1].rgb * sh1(n) +
        scene.gi_sh[2].rgb * sh2(n) +
        scene.gi_sh[3].rgb * sh3(n) +
        scene.gi_sh[4].rgb * sh4(n) +
        scene.gi_sh[5].rgb * sh5(n) +
        scene.gi_sh[6].rgb * sh6(n) +
        scene.gi_sh[7].rgb * sh7(n) +
        scene.gi_sh[8].rgb * sh8(n);

    return max(irradiance, vec3(0.0));
}

float probe_fade(vec3 world_pos) {
    float probe_radius = max(scene.gi_probe_pos_radius.w, 0.001);
    float distance_to_probe = length(world_pos - scene.gi_probe_pos_radius.xyz);

    return 1.0 - smoothstep(probe_radius * 0.75, probe_radius * 1.45, distance_to_probe);
}

vec2 environment_brdf_approx(vec3 f0, float roughness, float ndotv) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;

    return vec2(-1.04, 1.04) * a004 + r.zw;
}

GiSample sample_gi(
    MaterialSample material,
    vec3 f0,
    float ndotv,
    vec3 reflection,
    vec3 world_pos
) {
    vec3 diffuse_ibl = mix(
        environment_color(material.normal) * PI * 0.28,
        irradiance_probe(material.normal),
        saturate(probe_fade(world_pos) * scene.gi_params.x)
    );
    vec3 fresnel = fresnel_schlick_roughness(ndotv, f0, material.roughness);
    vec2 brdf = environment_brdf_approx(f0, material.roughness, ndotv);
    float energy = mix(0.35, 1.0, material.specular) * mix(1.0, 0.42, material.roughness);

    GiSample gi;
    gi.diffuse = diffuse_ibl * material.base.rgb * (1.0 - material.metallic) * material.ao / PI;
    gi.specular = environment_reflection(reflection, material.roughness, world_pos)
        * (fresnel * brdf.x + brdf.y)
        * material.ao
        * energy
        * scene.gi_params.y;

    return gi;
}

float sample_shadow_compare(vec2 uv, float compare_depth) {
    return compare_depth <= texture(shadow_map, uv).r ? 1.0 : 0.0;
}

int select_shadow_cascade(vec3 world_pos) {
    int count = int(clamp(scene.shadow_params.z, 0.0, float(SHADOW_CASCADE_COUNT)));
    float depth = max(dot(world_pos - scene.shadow_camera_pos.xyz, scene.shadow_camera_dir.xyz), 0.0);

    for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        if (i >= count) {
            break;
        }
        if (depth <= scene.shadow_cascade_params[i].x) {
            return i;
        }
    }

    return -1;
}

float shadow_visibility(vec3 world_pos, vec3 light, vec3 normal) {
    if (scene.shadow_params.w < 0.5 || scene.shadow_params.y <= 0.0) {
        return 1.0;
    }

    int cascade = select_shadow_cascade(world_pos);
    if (cascade < 0) {
        return 1.0;
    }

    vec4 projected = scene.shadow_view_proj[cascade] * vec4(world_pos, 1.0);
    if (projected.w <= 0.0) {
        return 1.0;
    }

    vec3 shadow_pos = projected.xyz / projected.w;
    vec2 local_uv = shadow_pos.xy * 0.5 + 0.5;
    if (
        local_uv.x <= 0.0 || local_uv.x >= 1.0 ||
        local_uv.y <= 0.0 || local_uv.y >= 1.0 ||
        shadow_pos.z <= 0.0 || shadow_pos.z >= 1.0
    ) {
        return 1.0;
    }

    vec4 atlas = scene.shadow_atlas[cascade];
    vec2 uv = atlas.xy + local_uv * atlas.zw;
    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
    vec2 atlas_min = atlas.xy + texel * 1.5;
    vec2 atlas_max = atlas.xy + atlas.zw - texel * 1.5;
    float light_alignment = saturate(dot(normalize(normal), light));
    float receiver_depth = shadow_pos.z
        - scene.shadow_cascade_params[cascade].y * mix(2.6, 0.9, light_alignment)
        - fwidth(shadow_pos.z) * 1.55;
    const vec2 poisson_disk[8] = vec2[](
        vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457), vec2(-0.203, 0.621),
        vec2(0.962, -0.195), vec2(0.473, -0.480), vec2(0.519, 0.767), vec2(0.185, -0.893)
    );

    int count = int(clamp(scene.shadow_params.z, 1.0, float(SHADOW_CASCADE_COUNT)));
    int sample_count = cascade <= 1 ? 8 : 5;
    float cascade_weight = float(cascade) / max(float(count - 1), 1.0);
    float kernel_radius = mix(1.05, 1.85, cascade_weight) * mix(1.18, 0.82, light_alignment);
    float lit = sample_shadow_compare(clamp(uv, atlas_min, atlas_max), receiver_depth) * 2.0;
    float weight_sum = 2.0;

    for (int i = 0; i < 8; i++) {
        if (i >= sample_count) {
            break;
        }
        float weight = (i < 4) ? 1.2 : 1.0;
        vec2 sample_uv = clamp(uv + poisson_disk[i] * texel * kernel_radius, atlas_min, atlas_max);
        lit += sample_shadow_compare(sample_uv, receiver_depth) * weight;
        weight_sum += weight;
    }

    return mix(1.0 - scene.shadow_params.y, 1.0, lit / weight_sum);
}

#endif
