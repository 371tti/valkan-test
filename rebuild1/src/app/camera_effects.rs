use crate::protocol::{CameraEffects, Exposure, FramebufferMetering};

#[derive(Clone, Copy, Debug)]
pub(super) struct CameraMetering {
    pub valid: bool,
    pub average_luminance: f32,
    pub center_luminance: f32,
    pub highlight_fraction: f32,
    pub average_color: [f32; 3],
    pub white_balance_confidence: f32,
}

#[derive(Clone, Debug)]
pub(super) struct CameraEffectController {
    exposure: f32,
    white_balance: [f32; 3],
}

const MIN_OLD_CAMERA_EXPOSURE: f32 = 0.35;
const MAX_CAMERA_EXPOSURE: f32 = 3.0;
const TARGET_CAMERA_LUMA: f32 = 0.20;
const EXPOSURE_RISE_STOPS_PER_SECOND: f32 = 1.1;
const EXPOSURE_FALL_STOPS_PER_SECOND: f32 = 3.0;
const MIN_WHITE_BALANCE: f32 = 0.55;
const MAX_WHITE_BALANCE: f32 = 1.85;
const OLD_CAMERA_CONTRAST: f32 = 1.06;
const OLD_CAMERA_SATURATION: f32 = 1.04;

impl Default for CameraMetering {
    /// Creates an invalid metering sample that leaves the current camera response unchanged.
    fn default() -> Self {
        Self {
            valid: false,
            average_luminance: 0.0,
            center_luminance: 0.0,
            highlight_fraction: 0.0,
            average_color: [0.0; 3],
            white_balance_confidence: 0.0,
        }
    }
}

impl CameraMetering {
    /// Creates the neutral screen sample used until real framebuffer readback arrives.
    pub(super) fn neutral_screen() -> Self {
        Self::estimated(0.18, 0.18, 0.0, [0.18; 3], 0.0)
    }

    /// Creates one app-side metering sample from estimated linear scene luminance.
    pub(super) fn estimated(
        average_luminance: f32,
        center_luminance: f32,
        highlight_fraction: f32,
        average_color: [f32; 3],
        white_balance_confidence: f32,
    ) -> Self {
        let valid_color = average_color.iter().all(|value| value.is_finite());
        if !average_luminance.is_finite()
            || !center_luminance.is_finite()
            || !highlight_fraction.is_finite()
            || !white_balance_confidence.is_finite()
            || !valid_color
        {
            return Self::default();
        }

        Self {
            valid: true,
            average_luminance: average_luminance.max(0.0),
            center_luminance: center_luminance.max(0.0),
            highlight_fraction: highlight_fraction.clamp(0.0, 1.0),
            average_color: average_color.map(|value| value.max(0.0)),
            white_balance_confidence: white_balance_confidence.clamp(0.0, 1.0),
        }
    }

    /// Copies renderer framebuffer metering into the old AutoCamera input shape.
    pub(super) fn from_framebuffer(metering: FramebufferMetering) -> Self {
        if !metering.valid() {
            return Self::default();
        }

        Self::estimated(
            metering.average_luminance(),
            metering.center_luminance(),
            metering.highlight_fraction(),
            metering.average_color(),
            metering.white_balance_confidence(),
        )
    }
}

impl Default for CameraEffectController {
    /// Creates a neutral response before the first metering sample is available.
    fn default() -> Self {
        Self {
            exposure: 1.0,
            white_balance: [1.0; 3],
        }
    }
}

impl CameraEffectController {
    /// Updates auto exposure and white balance from one app-side metering sample.
    ///
    /// The input must be old-style screen-space metering from the final displayed framebuffer.
    pub(super) fn update(&mut self, metering: CameraMetering, delta_time: f32) -> CameraEffects {
        let delta_time = delta_time.clamp(0.0, 0.25);

        if metering.valid {
            self.update_exposure(metering, delta_time);
            self.update_white_balance(metering, delta_time);
        }

        self.response()
    }

    /// Returns the current camera response without applying a new metering sample.
    fn response(&self) -> CameraEffects {
        let exposure = Exposure::new(
            self.exposure
                .clamp(MIN_OLD_CAMERA_EXPOSURE, MAX_CAMERA_EXPOSURE),
        )
        .expect("camera exposure controller keeps finite non-negative values");
        CameraEffects::with_look(
            exposure,
            clamp_white_balance(self.white_balance),
            OLD_CAMERA_CONTRAST,
            OLD_CAMERA_SATURATION,
            true,
        )
        .expect("camera effect controller keeps bounded white balance")
    }

    /// Moves exposure with the same screen-space response curve as the old AutoCamera.
    fn update_exposure(&mut self, metering: CameraMetering, delta_time: f32) {
        let metered_luma =
            (metering.center_luminance * 0.65 + metering.average_luminance * 0.35).max(0.002);
        let low_light = 1.0 - smoothstep(0.035, 0.22, metered_luma);
        let target_display_luma = smooth_mix(TARGET_CAMERA_LUMA, 0.26, low_light);
        let max_exposure = smooth_mix(2.2, MAX_CAMERA_EXPOSURE, low_light);
        let highlight_guard = (1.0_f32
            - (metering.highlight_fraction - 0.04_f32).max(0.0_f32) * 1.6_f32)
            .clamp(0.60_f32, 1.0_f32);
        let target_exposure =
            (self.exposure * (target_display_luma / metered_luma) * highlight_guard)
                .clamp(MIN_OLD_CAMERA_EXPOSURE, max_exposure);
        let exposure_seconds = if target_exposure < self.exposure {
            smooth_mix(0.15, 0.35, low_light)
        } else {
            smooth_mix(0.55, 1.4, low_light)
        };
        let smoothed =
            smooth_exposure(self.exposure, target_exposure, delta_time, exposure_seconds)
                .clamp(MIN_OLD_CAMERA_EXPOSURE, MAX_CAMERA_EXPOSURE);

        self.exposure = limit_exposure_step(self.exposure, smoothed, delta_time);
    }

    /// Moves per-channel white balance only when metering has enough chroma signal.
    fn update_white_balance(&mut self, metering: CameraMetering, delta_time: f32) {
        if metering.white_balance_confidence <= 0.08 {
            return;
        }

        let target_wb = white_balance_for_screen(metering, self.white_balance);
        let wb_seconds = 4.0 / metering.white_balance_confidence.clamp(0.25, 1.0);

        for (current, target) in self.white_balance.iter_mut().zip(target_wb.iter().copied()) {
            *current = smooth_exposure(*current, target, delta_time, wb_seconds)
                .clamp(MIN_WHITE_BALANCE, MAX_WHITE_BALANCE);
        }
    }
}

/// Computes a bounded white-balance correction from the metered average screen color.
fn white_balance_for_screen(metering: CameraMetering, current: [f32; 3]) -> [f32; 3] {
    let gray = luminance(metering.average_color).max(0.001);

    std::array::from_fn(|channel| {
        let correction = (gray / metering.average_color[channel].max(0.001)).clamp(0.72, 1.38);
        (current[channel] * correction.powf(0.45)).clamp(MIN_WHITE_BALANCE, MAX_WHITE_BALANCE)
    })
}

/// Clamps white balance into the renderer protocol's tighter camera-response range.
fn clamp_white_balance(value: [f32; 3]) -> [f32; 3] {
    value.map(|channel| channel.clamp(MIN_WHITE_BALANCE, MAX_WHITE_BALANCE))
}

/// Returns Rec. 709 luminance for linear RGB values.
fn luminance(color: [f32; 3]) -> f32 {
    color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
}

/// Smoothly interpolates with a bounded exponential response time.
fn smooth_toward(current: f32, target: f32, delta_time: f32, seconds: f32) -> f32 {
    let weight = 1.0 - (-delta_time / seconds.max(0.001)).exp();

    smooth_mix(current, target, weight)
}

/// Smooths exposure-like positive values in log space.
fn smooth_exposure(current: f32, target: f32, delta_time: f32, seconds: f32) -> f32 {
    smooth_toward(
        current.max(0.0001).ln(),
        target.max(0.0001).ln(),
        delta_time,
        seconds,
    )
    .exp()
}

/// Limits exposure change speed in photographic stops per second.
fn limit_exposure_step(current: f32, target: f32, delta_time: f32) -> f32 {
    let current = current
        .clamp(MIN_OLD_CAMERA_EXPOSURE, MAX_CAMERA_EXPOSURE)
        .max(0.0001);
    let rise = 2.0_f32.powf(EXPOSURE_RISE_STOPS_PER_SECOND * delta_time);
    let fall = 2.0_f32.powf(EXPOSURE_FALL_STOPS_PER_SECOND * delta_time);

    target
        .clamp(current / fall, current * rise)
        .clamp(MIN_OLD_CAMERA_EXPOSURE, MAX_CAMERA_EXPOSURE)
}

/// Returns a Hermite smoothstep value for bounded camera response blending.
fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(0.0001)).clamp(0.0, 1.0);

    t * t * (3.0 - 2.0 * t)
}

/// Linearly interpolates two scalar camera response values.
fn smooth_mix(a: f32, b: f32, weight: f32) -> f32 {
    a + (b - a) * weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_metering_stays_in_old_camera_bounds() {
        let mut controller = CameraEffectController::default();

        for _ in 0..180 {
            controller.update(metering(0.001, 0.0), 1.0 / 60.0);
        }

        assert!(controller.exposure <= MAX_CAMERA_EXPOSURE);
        assert!(controller.exposure >= MIN_OLD_CAMERA_EXPOSURE);
    }

    #[test]
    fn bright_metering_recovers_exposure_downward() {
        let mut controller = CameraEffectController {
            exposure: MAX_CAMERA_EXPOSURE,
            white_balance: [1.0; 3],
        };

        for _ in 0..8 {
            controller.update(metering(0.68, 0.18), 1.0 / 30.0);
        }

        assert!(controller.exposure < 1.8);
    }

    #[test]
    fn warm_metering_pushes_white_balance_cooler() {
        let mut controller = CameraEffectController::default();

        for _ in 0..180 {
            controller.update(
                CameraMetering::estimated(0.24, 0.26, 0.0, [0.33, 0.24, 0.13], 1.0),
                1.0 / 60.0,
            );
        }

        let white_balance = controller.response().white_balance();
        assert!(white_balance[2] > white_balance[0]);
    }

    #[test]
    fn white_balance_stays_bounded() {
        let mut controller = CameraEffectController {
            exposure: 1.0,
            white_balance: [0.1, 4.0, 1.0],
        };

        for _ in 0..180 {
            controller.update(
                CameraMetering::estimated(0.18, 0.18, 0.0, [0.30, 0.12, 0.08], 1.0),
                1.0 / 60.0,
            );
        }

        let effects = controller.response();
        assert!(effects.white_balance()[0] >= MIN_WHITE_BALANCE);
        assert!(effects.white_balance()[1] <= MAX_WHITE_BALANCE);
    }

    fn metering(luma: f32, highlight_fraction: f32) -> CameraMetering {
        CameraMetering::estimated(luma, luma, highlight_fraction, [luma; 3], 0.0)
    }
}
