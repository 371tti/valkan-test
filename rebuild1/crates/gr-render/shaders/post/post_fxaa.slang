#pragma once

float depth_edge_from_depths(float center_depth, float center_z, float sample_depth) {
    if (is_background_depth(center_depth) && is_background_depth(sample_depth)) {
        return 0.0;
    }

    float sample_z = linear_depth(sample_depth);

    float scale = max(min(center_z, sample_z) * 0.018, 0.015);
    return saturate(abs(center_z - sample_z) * rcp_safe(scale, 0.0001));
}

float normal_edge_from_depth(
    float3 center_normal,
    float center_depth,
    float sample_depth,
    float2 sample_uv
) {
    if (is_background_depth(center_depth) || is_background_depth(sample_depth)) {
        return 0.0;
    }

    float4 sample_meta = opaque_metadata_at(sample_uv);
    float3 sample_normal = oct_decode(sample_meta.xy);

    return saturate((1.0 - dot(center_normal, sample_normal)) * 2.4);
}

float3 high_quality_fxaa_scene_color(
    float2 uv,
    float3 center,
    SurfaceMaterial center_material,
    bool center_material_valid,
    bool aa_enabled
) {
    if (!aa_enabled) {
        return center;
    }

    float2 texel = params.aa.xy;

    float2 uv_n = uv + float2(0.0, -texel.y);
    float2 uv_s = uv + float2(0.0,  texel.y);
    float2 uv_w = uv + float2(-texel.x, 0.0);
    float2 uv_e = uv + float2( texel.x, 0.0);

    float3 north = sample_texture(scene_color, uv_n).rgb;
    float3 south = sample_texture(scene_color, uv_s).rgb;
    float3 west  = sample_texture(scene_color, uv_w).rgb;
    float3 east  = sample_texture(scene_color, uv_e).rgb;

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

    float2 uv_nw = uv + float2(-texel.x, -texel.y);
    float2 uv_ne = uv + float2( texel.x, -texel.y);
    float2 uv_sw = uv + float2(-texel.x,  texel.y);
    float2 uv_se = uv + float2( texel.x,  texel.y);

    float3 northwest = sample_texture(scene_color, uv_nw).rgb;
    float3 northeast = sample_texture(scene_color, uv_ne).rgb;
    float3 southwest = sample_texture(scene_color, uv_sw).rgb;
    float3 southeast = sample_texture(scene_color, uv_se).rgb;

    float luma_nw = perceptual_luma(northwest);
    float luma_ne = perceptual_luma(northeast);
    float luma_sw = perceptual_luma(southwest);
    float luma_se = perceptual_luma(southeast);

    luma_min = min(luma_min, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    luma_max = max(luma_max, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    float2 dir = float2(
        -((luma_nw + luma_ne) - (luma_sw + luma_se)),
         ((luma_nw + luma_sw) - (luma_ne + luma_se))
    );

    if (dot(dir, dir) <= 0.000001) {
        dir = geometry_vertical > geometry_horizontal
            ? float2(0.0, 1.0)
            : float2(1.0, 0.0);
    }

    float reduce = max(
        (luma_nw + luma_ne + luma_sw + luma_se) * 0.0078125,
        0.0009765625
    );

    float scale = rcp_safe(min(abs(dir.x), abs(dir.y)) + reduce, 0.000001);
    dir = clamp(dir * scale, float2(-8.0), float2(8.0)) * texel;

    float3 rgb_a = 0.5 * (
        sample_texture(scene_color, uv + dir * -0.16666667).rgb +
        sample_texture(scene_color, uv + dir *  0.16666667).rgb
    );

    float3 rgb_b = rgb_a * 0.5 + 0.25 * (
        sample_texture(scene_color, uv + dir * -0.5).rgb +
        sample_texture(scene_color, uv + dir *  0.5).rgb
    );

    float luma_b = perceptual_luma(rgb_b);
    float3 resolved = (luma_b < luma_min || luma_b > luma_max) ? rgb_a : rgb_b;
    float blend = saturate(
        (edge_signal - params.aa.z) * rcp_safe(edge_signal, 0.0001)
    ) * params.aa.w;

    return lerp(center, resolved, blend);
}
