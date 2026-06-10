use super::{FrameId, NonZeroExtent, SurfaceGeneration, SurfaceId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FramebufferReadbackOptions {
    enabled: bool,
    interval_frames: u32,
}

impl FramebufferReadbackOptions {
    /// Enables or disables app-visible framebuffer metering at a bounded frame interval.
    pub fn new(enabled: bool, interval_frames: u32) -> Self {
        Self {
            enabled,
            interval_frames: interval_frames.max(1),
        }
    }

    /// Requests the old AutoCamera-style final framebuffer metering cadence.
    pub fn camera_metering() -> Self {
        Self::new(true, 12)
    }

    /// Returns whether the renderer should copy framebuffer pixels for app metering.
    pub fn enabled(self) -> bool {
        self.enabled
    }

    /// Returns the minimum number of submitted frames between GPU readback copies.
    pub fn interval_frames(self) -> u32 {
        self.interval_frames
    }
}

impl Default for FramebufferReadbackOptions {
    /// Keeps framebuffer readback disabled unless the app explicitly asks for it.
    fn default() -> Self {
        Self::new(false, 3)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramebufferMetering {
    valid: bool,
    average_luminance: f32,
    center_luminance: f32,
    highlight_fraction: f32,
    average_color: [f32; 3],
    white_balance_confidence: f32,
}

impl FramebufferMetering {
    /// Creates a bounded luminance and color summary from renderer-owned framebuffer pixels.
    pub fn new(
        average_luminance: f32,
        center_luminance: f32,
        highlight_fraction: f32,
        average_color: [f32; 3],
        white_balance_confidence: f32,
    ) -> Self {
        let valid_color = average_color.iter().all(|value| value.is_finite());
        let valid_scalars = [
            average_luminance,
            center_luminance,
            highlight_fraction,
            white_balance_confidence,
        ]
        .iter()
        .all(|value| value.is_finite());

        if !valid_color || !valid_scalars {
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

    /// Returns whether this event contains a usable framebuffer sample.
    pub fn valid(self) -> bool {
        self.valid
    }

    /// Returns center-weighted average luminance measured from the displayed image.
    pub fn average_luminance(self) -> f32 {
        self.average_luminance
    }

    /// Returns the central image luminance used by auto exposure.
    pub fn center_luminance(self) -> f32 {
        self.center_luminance
    }

    /// Returns the weighted fraction of samples near display white.
    pub fn highlight_fraction(self) -> f32 {
        self.highlight_fraction
    }

    /// Returns the center-weighted average linear RGB display color.
    pub fn average_color(self) -> [f32; 3] {
        self.average_color
    }

    /// Returns whether the average color is useful for automatic white balance.
    pub fn white_balance_confidence(self) -> f32 {
        self.white_balance_confidence
    }
}

impl Default for FramebufferMetering {
    /// Creates an invalid sample that app-side camera controllers can ignore.
    fn default() -> Self {
        Self {
            valid: false,
            average_luminance: 0.18,
            center_luminance: 0.18,
            highlight_fraction: 0.0,
            average_color: [0.18; 3],
            white_balance_confidence: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramebufferReadback {
    pub frame_id: FrameId,
    pub surface_id: SurfaceId,
    pub generation: SurfaceGeneration,
    pub extent: NonZeroExtent,
    pub metering: FramebufferMetering,
}

impl FramebufferReadback {
    /// Packages one renderer-owned framebuffer metering result for app-side camera effects.
    pub fn new(
        frame_id: FrameId,
        surface_id: SurfaceId,
        generation: SurfaceGeneration,
        extent: NonZeroExtent,
        metering: FramebufferMetering,
    ) -> Self {
        Self {
            frame_id,
            surface_id,
            generation,
            extent,
            metering,
        }
    }
}
