#ifndef REBUILD1_POST_FXAA_GLSL
#define REBUILD1_POST_FXAA_GLSL

float depth_edge_from_depths(float center_depth, float center_z, float sample_depth) {
    if (is_background_depth(center_depth) && is_background_depth(sample_depth)) {
        return 0.0;
    }

    float sample_z = linear_depth(sample_depth);

    float scale = max(min(center_z, sample_z) * 0.018, 0.015);
    return saturate(abs(center_z - sample_z) * rcp_safe(scale, 0.0001));
}

float normal_edge_from_depth(
    vec3 center_normal,
    float center_depth,
    float sample_depth,
    vec2 sample_uv
) {
    if (is_background_depth(center_depth) || is_background_depth(sample_depth)) {
        return 0.0;
    }

    vec4 sample_meta = texture(scene_normal_roughness, sample_uv);
    vec3 sample_normal = oct_decode(sample_meta.xy);

    return saturate((1.0 - dot(center_normal, sample_normal)) * 2.4);
}

vec3 high_quality_fxaa_scene_color(
    vec2 uv,
    vec3 center,
    SurfaceMaterial center_material,
    bool center_material_valid,
    bool aa_enabled
) {
    if (!aa_enabled) {
        return center;
    }

    vec2 texel = params.aa.xy;

    vec2 uv_n = uv + vec2(0.0, -texel.y);
    vec2 uv_s = uv + vec2(0.0,  texel.y);
    vec2 uv_w = uv + vec2(-texel.x, 0.0);
    vec2 uv_e = uv + vec2( texel.x, 0.0);

    vec3 north = texture(scene_color, uv_n).rgb;
    vec3 south = texture(scene_color, uv_s).rgb;
    vec3 west  = texture(scene_color, uv_w).rgb;
    vec3 east  = texture(scene_color, uv_e).rgb;

    float luma_m = perceptual_luma(center);
    float luma_n = perceptual_luma(north);
    float luma_s = perceptual_luma(south);
    float luma_w = perceptual_luma(west);
    float luma_e = perceptual_luma(east);

    float luma_min = min(luma_m, min(min(luma_n, luma_s), min(luma_w, luma_e)));
    float luma_max = max(luma_m, max(max(luma_n, luma_s), max(luma_w, luma_e)));
    float edge_signal = luma_max - luma_min;

    float geometry_horizontal = 0.0;
    float geometry_vertical = 0.0;

    if (edge_signal <= params.aa.z) {
        SurfaceMaterial material = center_material;

        if (!center_material_valid) {
            material = surface_material(uv);
        }

        float depth_n = depth_at(uv_n);
        float depth_s = depth_at(uv_s);
        float depth_w = depth_at(uv_w);
        float depth_e = depth_at(uv_e);
        float center_z = linear_depth(material.source_depth);

        float depth_edge_n = depth_edge_from_depths(material.source_depth, center_z, depth_n);
        float depth_edge_s = depth_edge_from_depths(material.source_depth, center_z, depth_s);
        float depth_edge_w = depth_edge_from_depths(material.source_depth, center_z, depth_w);
        float depth_edge_e = depth_edge_from_depths(material.source_depth, center_z, depth_e);

        geometry_horizontal = max(depth_edge_n, depth_edge_s);
        geometry_vertical = max(depth_edge_w, depth_edge_e);

        float depth_signal = max(
            max(depth_edge_n, depth_edge_s),
            max(depth_edge_w, depth_edge_e)
        );

        float normal_signal = 0.0;

        if (depth_signal < 1.0) {
            float normal_edge_n = normal_edge_from_depth(
                material.normal,
                material.source_depth,
                depth_n,
                uv_n
            );
            float normal_edge_s = normal_edge_from_depth(
                material.normal,
                material.source_depth,
                depth_s,
                uv_s
            );
            float normal_edge_w = normal_edge_from_depth(
                material.normal,
                material.source_depth,
                depth_w,
                uv_w
            );
            float normal_edge_e = normal_edge_from_depth(
                material.normal,
                material.source_depth,
                depth_e,
                uv_e
            );

            normal_signal = max(
                max(normal_edge_n, normal_edge_s),
                max(normal_edge_w, normal_edge_e)
            );
        }

        edge_signal = max(edge_signal, max(depth_signal, normal_signal) * 0.50);
    }

    if (edge_signal <= params.aa.z) {
        return center;
    }

    vec2 uv_nw = uv + vec2(-texel.x, -texel.y);
    vec2 uv_ne = uv + vec2( texel.x, -texel.y);
    vec2 uv_sw = uv + vec2(-texel.x,  texel.y);
    vec2 uv_se = uv + vec2( texel.x,  texel.y);

    vec3 northwest = texture(scene_color, uv_nw).rgb;
    vec3 northeast = texture(scene_color, uv_ne).rgb;
    vec3 southwest = texture(scene_color, uv_sw).rgb;
    vec3 southeast = texture(scene_color, uv_se).rgb;

    float luma_nw = perceptual_luma(northwest);
    float luma_ne = perceptual_luma(northeast);
    float luma_sw = perceptual_luma(southwest);
    float luma_se = perceptual_luma(southeast);

    luma_min = min(luma_min, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    luma_max = max(luma_max, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    vec2 dir = vec2(
        -((luma_nw + luma_ne) - (luma_sw + luma_se)),
         ((luma_nw + luma_sw) - (luma_ne + luma_se))
    );

    if (dot(dir, dir) <= 0.000001) {
        dir = geometry_vertical > geometry_horizontal
            ? vec2(0.0, 1.0)
            : vec2(1.0, 0.0);
    }

    float reduce = max(
        (luma_nw + luma_ne + luma_sw + luma_se) * 0.0078125,
        0.0009765625
    );

    float scale = rcp_safe(min(abs(dir.x), abs(dir.y)) + reduce, 0.000001);
    dir = clamp(dir * scale, vec2(-8.0), vec2(8.0)) * texel;

    vec3 rgb_a = 0.5 * (
        texture(scene_color, uv + dir * -0.16666667).rgb +
        texture(scene_color, uv + dir *  0.16666667).rgb
    );

    vec3 rgb_b = rgb_a * 0.5 + 0.25 * (
        texture(scene_color, uv + dir * -0.5).rgb +
        texture(scene_color, uv + dir *  0.5).rgb
    );

    float luma_b = perceptual_luma(rgb_b);
    vec3 resolved = (luma_b < luma_min || luma_b > luma_max) ? rgb_a : rgb_b;
    float blend = saturate(
        (edge_signal - params.aa.z) * rcp_safe(edge_signal, 0.0001)
    ) * params.aa.w;

    return mix(center, resolved, blend);
}

#endif
