#pragma once

// SMAA 1x shared implementation.
//
// This file deliberately contains the equations shared by the three fullscreen passes:
//   1. luma edge detection writes edgesTex (RG)
//   2. blending-weight calculation reads edgesTex and writes blendTex (RGBA)
//   3. neighbourhood blending reads blendTex and writes the presentation target
//
// The layout follows the reference implementation in iryoku/smaa. In particular, the edge
// search samples edgesTex with a linear sampler; replacing those reads with point samples creates
// the dotted edge/weight pattern that this renderer used to expose.

[[vk::binding(0, 2)]] Sampler2D final_color;
[[vk::binding(1, 2)]] Sampler2D edges_tex;
[[vk::binding(2, 2)]] Sampler2D weights_tex;
[[vk::binding(3, 2)]] Sampler2D area_tex;
[[vk::binding(4, 2)]] Sampler2D search_tex;

struct SmaaParams {
    // xy = inverse render-target dimensions, z = luma threshold, w = resolve blend strength.
    float4 aa;
    // Retained for push-constant ABI stability with the renderer's existing diagnostics.
    float4 depth;
    // x = DebugViewMode numeric value, y = explicit SMAA ON/OFF switch.
    float4 debug;
};
[[vk::push_constant]] ConstantBuffer<SmaaParams> params;

static const int SMAA_MAX_SEARCH_STEPS = 16;
static const int SMAA_MAX_SEARCH_STEPS_DIAG = 8;
static const float SMAA_CORNER_ROUNDING_NORM = 0.25;
static const float SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR = 2.0;
static const float SMAA_AREATEX_MAX_DISTANCE = 16.0;
static const float SMAA_AREATEX_MAX_DISTANCE_DIAG = 20.0;
static const float SMAA_AREATEX_SUBTEX_SIZE = 1.0 / 7.0;
static const float2 SMAA_AREATEX_PIXEL_SIZE = float2(1.0 / 160.0, 1.0 / 560.0);
static const float2 SMAA_SEARCHTEX_SIZE = float2(66.0, 33.0);
static const float2 SMAA_SEARCHTEX_PACKED_SIZE = float2(64.0, 16.0);

float2 smaa_clamp_uv(float2 uv) {
    return clamp(uv, float2(0.0), float2(1.0));
}

bool smaa_enabled() {
    return params.debug.y > 0.5;
}

int2 smaa_pixel_coordinate(float2 uv) {
    int2 dimensions = max(texture_dimensions(final_color, 0), int2(1, 1));
    int2 maximum = max(dimensions - int2(1, 1), int2(0, 0));
    int2 coordinate = int2(floor(smaa_clamp_uv(uv) * float2(dimensions)));
    return min(max(coordinate, int2(0, 0)), maximum);
}

// The reference edge pass uses point taps. PostColor is linear FP16, so encode at the luma
// boundary instead of making the edge threshold depend on HDR scene exposure.
float3 smaa_color_point(float2 uv) {
    return load_texel(final_color, smaa_pixel_coordinate(uv), 0).rgb;
}

float3 smaa_color_linear(float2 uv) {
    // The reference resolve uses an explicit mip-zero sample. Keeping the LOD explicit also
    // avoids derivative-dependent filtering in the divergent neighbourhood branch.
    return sample_texture_level(final_color, smaa_clamp_uv(uv), 0.0).rgb;
}

float smaa_srgb_encode(float value) {
    value = saturate(value);
    return value <= 0.0031308
        ? 12.92 * value
        : 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

float smaa_luma(float3 color) {
    float3 display_color = float3(
        smaa_srgb_encode(max(color.r, 0.0)),
        smaa_srgb_encode(max(color.g, 0.0)),
        smaa_srgb_encode(max(color.b, 0.0))
    );
    return dot(display_color, float3(0.2126, 0.7152, 0.0722));
}

// Pass 1: luma edge detection. R is the west/left edge and G is the north/top edge, matching
// SMAALumaEdgeDetectionPS and the channel convention used by the search pass.
float2 smaa_detect_edges(float2 uv) {
    float2 texel = params.aa.xy;
    float center = smaa_luma(smaa_color_point(uv));
    float left = smaa_luma(smaa_color_point(uv - float2(texel.x, 0.0)));
    float top = smaa_luma(smaa_color_point(uv - float2(0.0, texel.y)));
    float right = smaa_luma(smaa_color_point(uv + float2(texel.x, 0.0)));
    float bottom = smaa_luma(smaa_color_point(uv + float2(0.0, texel.y)));

    float2 delta = abs(center - float2(left, top));
    float2 edges = step(max(params.aa.zz, float2(0.000001)), delta);
    if (dot(edges, float2(1.0, 1.0)) == 0.0) {
        return float2(0.0);
    }

    float2 direct_delta = abs(center - float2(right, bottom));
    float left_left = smaa_luma(smaa_color_point(uv - float2(2.0 * texel.x, 0.0)));
    float top_top = smaa_luma(smaa_color_point(uv - float2(0.0, 2.0 * texel.y)));
    float2 max_delta = max(delta, direct_delta);
    float2 far_delta = abs(float2(left, top) - float2(left_left, top_top));
    max_delta = max(max_delta, far_delta);
    float final_delta = max(max_delta.x, max_delta.y);
    edges *= step(final_delta, SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta);
    return saturate(edges);
}

// All edge texture reads in the weight pass are intentionally linear and clamped. The texture
// is written as R8G8_UNORM but never sampled as sRGB.
float2 smaa_edge_sample(float2 uv) {
    // SMAASearchX/Y and the diagonal loops are divergent. The reference implementation uses
    // level-zero linear reads there; an implicit-LOD sample can make a one-pixel edge alternate
    // between search hits on drivers that evaluate derivatives per branch.
    return sample_texture_level(edges_tex, smaa_clamp_uv(uv), 0.0).rg;
}

float smaa_search_length(float2 edge, float offset) {
    float2 scale = SMAA_SEARCHTEX_SIZE * float2(0.5, -1.0);
    float2 bias = SMAA_SEARCHTEX_SIZE * float2(offset, 1.0);
    scale += float2(-1.0, 1.0);
    bias += float2(0.5, -0.5);
    scale *= 1.0 / SMAA_SEARCHTEX_PACKED_SIZE;
    bias *= 1.0 / SMAA_SEARCHTEX_PACKED_SIZE;
    return sample_texture_level(search_tex,
        clamp(mad(scale, edge, bias), float2(0.0), float2(1.0)), 0.0).r;
}

float smaa_search_x_left(float2 coordinate, float end) {
    float2 texel = params.aa.xy;
    float2 edge = float2(0.0, 1.0);
    while (coordinate.x > end && edge.g > 0.8281 && edge.r == 0.0) {
        edge = smaa_edge_sample(coordinate);
        coordinate -= float2(2.0 * texel.x, 0.0);
    }
    float offset = 3.25 - (255.0 / 127.0) * smaa_search_length(edge, 0.0);
    return mad(texel.x, offset, coordinate.x);
}

float smaa_search_x_right(float2 coordinate, float end) {
    float2 texel = params.aa.xy;
    float2 edge = float2(0.0, 1.0);
    while (coordinate.x < end && edge.g > 0.8281 && edge.r == 0.0) {
        edge = smaa_edge_sample(coordinate);
        coordinate += float2(2.0 * texel.x, 0.0);
    }
    float offset = 3.25 - (255.0 / 127.0) * smaa_search_length(edge, 0.5);
    return mad(-texel.x, offset, coordinate.x);
}

float smaa_search_y_up(float2 coordinate, float end) {
    float2 texel = params.aa.xy;
    float2 edge = float2(1.0, 0.0);
    while (coordinate.y > end && edge.r > 0.8281 && edge.g == 0.0) {
        edge = smaa_edge_sample(coordinate);
        coordinate -= float2(0.0, 2.0 * texel.y);
    }
    float offset = 3.25 - (255.0 / 127.0) * smaa_search_length(edge.gr, 0.0);
    return mad(texel.y, offset, coordinate.y);
}

float smaa_search_y_down(float2 coordinate, float end) {
    float2 texel = params.aa.xy;
    float2 edge = float2(1.0, 0.0);
    while (coordinate.y < end && edge.r > 0.8281 && edge.g == 0.0) {
        edge = smaa_edge_sample(coordinate);
        coordinate += float2(0.0, 2.0 * texel.y);
    }
    float offset = 3.25 - (255.0 / 127.0) * smaa_search_length(edge.gr, 0.5);
    return mad(-texel.y, offset, coordinate.y);
}

float2 smaa_area(float2 distance, float edge_1, float edge_2) {
    float2 texture_coordinate = mad(
        float2(SMAA_AREATEX_MAX_DISTANCE, SMAA_AREATEX_MAX_DISTANCE),
        round(4.0 * float2(edge_1, edge_2)),
        distance
    );
    texture_coordinate = mad(
        SMAA_AREATEX_PIXEL_SIZE,
        texture_coordinate,
        0.5 * SMAA_AREATEX_PIXEL_SIZE
    );
    // Subsample index is zero for SMAA 1x.
    return sample_texture_level(area_tex, clamp(texture_coordinate,
        float2(0.0), float2(1.0)), 0.0).rg;
}

float2 smaa_decode_diag_bilinear(float2 edge) {
    edge.r = edge.r * abs(5.0 * edge.r - 3.75);
    return round(edge);
}

float4 smaa_decode_diag_bilinear(float4 edge) {
    edge.rb = edge.rb * abs(5.0 * edge.rb - 3.75);
    return round(edge);
}

float2 smaa_search_diag1(float2 coordinate, float2 direction, out float2 edge) {
    float4 state = float4(coordinate, -1.0, 1.0);
    edge = float2(0.0);
    for (int i = 0; i < SMAA_MAX_SEARCH_STEPS_DIAG; ++i) {
        if (state.z >= float(SMAA_MAX_SEARCH_STEPS_DIAG - 1) || state.w <= 0.9) {
            break;
        }
        state.xy += params.aa.xy * direction;
        state.z += 1.0;
        edge = smaa_edge_sample(state.xy);
        state.w = dot(edge, float2(0.5));
    }
    return state.zw;
}

float2 smaa_search_diag2(float2 coordinate, float2 direction, out float2 edge) {
    float4 state = float4(coordinate + float2(0.25 * params.aa.x, 0.0), -1.0, 1.0);
    edge = float2(0.0);
    for (int i = 0; i < SMAA_MAX_SEARCH_STEPS_DIAG; ++i) {
        if (state.z >= float(SMAA_MAX_SEARCH_STEPS_DIAG - 1) || state.w <= 0.9) {
            break;
        }
        state.xy += params.aa.xy * direction;
        state.z += 1.0;
        edge = smaa_decode_diag_bilinear(smaa_edge_sample(state.xy));
        state.w = dot(edge, float2(0.5));
    }
    return state.zw;
}

float2 smaa_area_diag(float2 distance, float2 edge) {
    float2 texture_coordinate = mad(
        float2(SMAA_AREATEX_MAX_DISTANCE_DIAG),
        edge,
        distance
    );
    texture_coordinate = mad(
        SMAA_AREATEX_PIXEL_SIZE,
        texture_coordinate,
        0.5 * SMAA_AREATEX_PIXEL_SIZE
    );
    texture_coordinate.x += 0.5;
    return sample_texture_level(area_tex, clamp(texture_coordinate,
        float2(0.0), float2(1.0)), 0.0).gr;
}

// Reference diagonal search. A non-zero result takes priority over the cardinal searches for
// this pixel, exactly like SMAACalculateDiagWeights in the paper implementation.
float2 smaa_calculate_diag_weights(float2 uv, float2 edge) {
    float2 texel = params.aa.xy;
    float2 weights = float2(0.0);
    float2 end;
    float4 distance;
    if (edge.r > 0.0) {
        distance.xz = smaa_search_diag1(uv, float2(-1.0, 1.0), end);
        distance.x += step(0.9, end.y);
    } else {
        distance.xz = float2(0.0);
    }
    distance.yw = smaa_search_diag1(uv, float2(1.0, -1.0), end);
    if (distance.x + distance.y > 2.0) {
        float4 coordinates = uv.xyxy + texel.xyxy *
            float4(-distance.x + 0.25, distance.x, distance.y, -distance.y - 0.25);
        float4 crossing;
        crossing.xy = smaa_edge_sample(coordinates.xy - float2(texel.x, 0.0));
        crossing.zw = smaa_edge_sample(coordinates.zw + float2(texel.x, 0.0));
        crossing.yxwz = smaa_decode_diag_bilinear(crossing);
        float2 merged = 2.0 * crossing.xz + crossing.yw;
        merged *= step(float2(0.9), distance.zw);
        weights += smaa_area_diag(distance.xy, merged);
    }

    distance.xz = smaa_search_diag2(uv, float2(-1.0, -1.0), end);
    if (smaa_edge_sample(uv + float2(texel.x, 0.0)).r > 0.0) {
        distance.yw = smaa_search_diag2(uv, float2(1.0, 1.0), end);
        distance.y += step(0.9, end.y);
    } else {
        distance.yw = float2(0.0);
    }
    if (distance.x + distance.y > 2.0) {
        float4 coordinates = uv.xyxy + texel.xyxy *
            float4(-distance.x, -distance.x, distance.y, distance.y);
        float4 crossing;
        crossing.x = smaa_edge_sample(coordinates.xy - float2(texel.x, 0.0)).g;
        crossing.y = smaa_edge_sample(coordinates.xy + float2(0.0, -texel.y)).r;
        crossing.zw = smaa_edge_sample(coordinates.zw + float2(texel.x, 0.0));
        float2 merged = 2.0 * crossing.xz + crossing.yw;
        merged *= step(float2(0.9), distance.zw);
        weights += smaa_area_diag(distance.xy, merged);
    }
    return saturate(weights);
}

void smaa_horizontal_corner_pattern(inout float2 weights, float4 coordinates, float2 distance) {
    float2 left_right = step(distance.xy, distance.yx);
    float2 rounding = (1.0 - SMAA_CORNER_ROUNDING_NORM) * left_right;
    rounding /= max(left_right.x + left_right.y, 1.0);
    float2 factor = float2(1.0);
    factor.x -= rounding.x * smaa_edge_sample(coordinates.xy + float2(0.0, params.aa.y)).r;
    factor.x -= rounding.y * smaa_edge_sample(coordinates.zw + params.aa.xy).r;
    factor.y -= rounding.x * smaa_edge_sample(coordinates.xy - float2(0.0, 2.0 * params.aa.y)).r;
    factor.y -= rounding.y * smaa_edge_sample(coordinates.zw + float2(params.aa.x, -2.0 * params.aa.y)).r;
    weights *= saturate(factor);
}

void smaa_vertical_corner_pattern(inout float2 weights, float4 coordinates, float2 distance) {
    float2 left_right = step(distance.xy, distance.yx);
    float2 rounding = (1.0 - SMAA_CORNER_ROUNDING_NORM) * left_right;
    rounding /= max(left_right.x + left_right.y, 1.0);
    float2 factor = float2(1.0);
    factor.x -= rounding.x * smaa_edge_sample(coordinates.xy + float2(params.aa.x, 0.0)).g;
    factor.x -= rounding.y * smaa_edge_sample(coordinates.zw + params.aa.xy).g;
    factor.y -= rounding.x * smaa_edge_sample(coordinates.xy + float2(-2.0 * params.aa.x, 0.0)).g;
    factor.y -= rounding.y * smaa_edge_sample(coordinates.zw + float2(-2.0 * params.aa.x, params.aa.y)).g;
    weights *= saturate(factor);
}

// Pass 2: line searches and AreaTex lookup. The endpoint/crossing coordinates intentionally use
// the same quarter-pixel offsets as SMAABlendingWeightCalculationVS.
float4 smaa_calculate_weights(float2 uv) {
    float2 texel = params.aa.xy;
    float2 pixel = uv / max(texel, float2(0.000001));
    float4 offset0 = uv.xyxy + texel.xyxy * float4(-0.25, -0.125, 1.25, -0.125);
    float4 offset1 = uv.xyxy + texel.xyxy * float4(-0.125, -0.25, -0.125, 1.25);
    float4 ends = float4(
        offset0.x - 2.0 * texel.x * float(SMAA_MAX_SEARCH_STEPS),
        offset0.z + 2.0 * texel.x * float(SMAA_MAX_SEARCH_STEPS),
        offset1.y - 2.0 * texel.y * float(SMAA_MAX_SEARCH_STEPS),
        offset1.w + 2.0 * texel.y * float(SMAA_MAX_SEARCH_STEPS)
    );
    float2 e = smaa_edge_sample(uv);
    float4 weights = float4(0.0);
    bool diagonal = false;

    if (e.g > 0.0) {
        float2 diagonal_weights = smaa_calculate_diag_weights(uv, e);
        if (dot(diagonal_weights, float2(1.0)) > 1e-5) {
            weights.rg = diagonal_weights;
            diagonal = true;
        }
    }

    if (e.g > 0.0 && !diagonal) {
        float4 coordinates = float4(0.0);
        coordinates.x = smaa_search_x_left(offset0.xy, ends.x);
        coordinates.y = offset1.y;
        float left_crossing = smaa_edge_sample(coordinates.xy).r;
        coordinates.z = smaa_search_x_right(offset0.zw, ends.y);
        float2 distance = abs(round(float2(coordinates.x, coordinates.z) / texel.x - pixel.xx));
        float right_crossing = smaa_edge_sample(float2(coordinates.z + texel.x, coordinates.y)).r;
        weights.rg = smaa_area(sqrt(distance), left_crossing, right_crossing);
        smaa_horizontal_corner_pattern(weights.rg, coordinates.xyzy, distance);
    }

    if (e.r > 0.0 && !diagonal) {
        float4 coordinates = float4(0.0);
        coordinates.y = smaa_search_y_up(offset1.xy, ends.z);
        coordinates.x = offset0.x;
        float top_crossing = smaa_edge_sample(coordinates.xy).g;
        coordinates.z = smaa_search_y_down(offset1.zw, ends.w);
        float2 distance = abs(round(float2(coordinates.y, coordinates.z) / texel.y - pixel.yy));
        float bottom_crossing = smaa_edge_sample(float2(coordinates.x, coordinates.z + texel.y)).g;
        weights.ba = smaa_area(sqrt(distance), top_crossing, bottom_crossing);
        smaa_vertical_corner_pattern(weights.ba, coordinates.xyxz, distance);
    }
    return saturate(weights);
}

// Pass 3: exact neighbourhood selection from SMAANeighborhoodBlendingPS. This is deliberately a
// center-preserving weighted resolve; there is no pure-neighbour average that can widen a 1px
// feature into a 3px smear.
float3 smaa_neighborhood_color(float2 uv, bool enabled) {
    float3 center = smaa_color_linear(uv);
    if (!enabled) {
        return center;
    }

    float2 texel = params.aa.xy;
    float4 a;
    a.x = sample_texture(weights_tex, smaa_clamp_uv(uv + float2(texel.x, 0.0))).a;
    a.y = sample_texture(weights_tex, smaa_clamp_uv(uv + float2(0.0, texel.y))).g;
    float4 current = sample_texture(weights_tex, smaa_clamp_uv(uv));
    a.wz = current.xz;
    if (dot(a, float4(1.0)) < 1e-5) {
        return center;
    }

    bool horizontal = max(a.x, a.z) > max(a.y, a.w);
    float4 blending_offset = horizontal
        ? float4(a.x, 0.0, a.z, 0.0)
        : float4(0.0, a.y, 0.0, a.w);
    float2 blending_weight = horizontal ? a.xz : a.yw;
    blending_weight /= max(dot(blending_weight, float2(1.0)), 1e-5);
    float4 blending_coordinate = mad(
        blending_offset,
        float4(texel.xy, -texel.xy),
        uv.xyxy
    );
    float3 resolved = blending_weight.x * smaa_color_linear(blending_coordinate.xy);
    resolved += blending_weight.y * smaa_color_linear(blending_coordinate.zw);
    return lerp(center, resolved, saturate(params.aa.w));
}
