#ifndef REBUILD1_SHADOW_ALPHA_GLSL
#define REBUILD1_SHADOW_ALPHA_GLSL

const uint ALPHA_MODE_CUTOUT = 1u;
const uint ALPHA_MODE_TRANSPARENT = 2u;

void discard_opaque_shadow_alpha(uint alpha_mode, float alpha_cutoff, float alpha) {
    if (alpha_mode == ALPHA_MODE_CUTOUT && alpha <= alpha_cutoff) {
        discard;
    }
    if (alpha_mode == ALPHA_MODE_TRANSPARENT) {
        discard;
    }
}

void discard_translucent_shadow_alpha(uint alpha_mode, float alpha) {
    if (alpha_mode != ALPHA_MODE_TRANSPARENT || alpha <= 0.001) {
        discard;
    }
}

#endif
