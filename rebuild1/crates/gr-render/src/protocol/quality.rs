#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderQualitySettings {
    ssao: SsaoQualitySettings,
    anti_aliasing: AntiAliasingQualitySettings,
    post: PostQualitySettings,
}

impl RenderQualitySettings {
    /// Creates a complete renderer quality profile for fullscreen post effects.
    pub fn new(
        ssao: SsaoQualitySettings,
        anti_aliasing: AntiAliasingQualitySettings,
        post: PostQualitySettings,
    ) -> Self {
        Self {
            ssao,
            anti_aliasing,
            post,
        }
    }

    /// Returns the lightest profile intended for editing and camera navigation.
    pub fn performance() -> Self {
        Self::new(
            SsaoQualitySettings::disabled(),
            AntiAliasingQualitySettings::disabled(),
            PostQualitySettings::natural(),
        )
    }

    /// Returns the interactive profile used when the app does not override renderer quality.
    pub fn interactive() -> Self {
        Self::new(
            SsaoQualitySettings::interactive(),
            AntiAliasingQualitySettings::interactive(),
            PostQualitySettings::natural(),
        )
    }

    /// Returns the balanced profile for scenes that can spend more post-process budget.
    pub fn balanced() -> Self {
        Self::new(
            SsaoQualitySettings::balanced(),
            AntiAliasingQualitySettings::balanced(),
            PostQualitySettings::natural(),
        )
    }

    /// Returns the expensive profile used when visual inspection matters more than frame time.
    pub fn high_quality() -> Self {
        Self::new(
            SsaoQualitySettings::balanced(),
            AntiAliasingQualitySettings::high_quality(),
            PostQualitySettings::natural(),
        )
    }

    /// Returns the screen-space ambient occlusion quality applied by the post pass.
    pub fn ssao(self) -> SsaoQualitySettings {
        self.ssao
    }

    /// Returns the post-pass antialiasing quality applied before tone mapping.
    pub fn anti_aliasing(self) -> AntiAliasingQualitySettings {
        self.anti_aliasing
    }

    /// Returns the final look multipliers applied after app-side camera effects.
    pub fn post(self) -> PostQualitySettings {
        self.post
    }
}

impl Default for RenderQualitySettings {
    /// Uses the renderer's balanced quality profile for the default visual path.
    fn default() -> Self {
        Self::balanced()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsaoQualitySettings {
    intensity: f32,
    radius: f32,
    bias: f32,
    sample_count: u32,
}

impl SsaoQualitySettings {
    /// Creates bounded SSAO controls used to keep indirect darkening predictable.
    pub fn new(intensity: f32, radius: f32, bias: f32, sample_count: u32) -> Self {
        Self {
            intensity: finite_clamp(intensity, 0.0, 1.0, 0.32),
            radius: finite_clamp(radius, 0.05, 2.5, 0.60),
            bias: finite_clamp(bias, 0.002, 0.08, 0.030),
            sample_count: sample_count.clamp(1, 8),
        }
    }

    /// Disables SSAO so the post pass can skip per-pixel occlusion samples.
    pub fn disabled() -> Self {
        Self::new(0.0, 0.50, 0.032, 1)
    }

    /// Returns an SSAO profile for normal interactive camera movement.
    pub fn interactive() -> Self {
        Self::new(0.16, 0.50, 0.032, 2)
    }

    /// Returns an SSAO profile that adds contact depth without crushing shaded surfaces.
    pub fn balanced() -> Self {
        Self::new(0.34, 0.65, 0.030, 4)
    }

    /// Returns the old inspection-quality SSAO profile.
    pub fn high_quality() -> Self {
        Self::new(0.48, 0.75, 0.028, 6)
    }

    /// Returns the global SSAO darkening strength.
    pub fn intensity(self) -> f32 {
        self.intensity
    }

    /// Returns the view-space sampling radius.
    pub fn radius(self) -> f32 {
        self.radius
    }

    /// Returns the depth bias that prevents flat surfaces from self-occluding.
    pub fn bias(self) -> f32 {
        self.bias
    }

    /// Returns how many fixed kernel samples the shader should evaluate.
    pub fn sample_count(self) -> u32 {
        self.sample_count
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AntiAliasingQualitySettings {
    edge_threshold: f32,
    blend: f32,
}

impl AntiAliasingQualitySettings {
    /// Creates bounded post AA controls for the high-quality FXAA resolve.
    pub fn new(edge_threshold: f32, blend: f32) -> Self {
        Self {
            edge_threshold: finite_clamp(edge_threshold, 0.004, 0.08, 0.028),
            blend: finite_clamp(blend, 0.0, 1.0, 0.78),
        }
    }

    /// Disables post AA for the lowest-latency editing profile.
    pub fn disabled() -> Self {
        Self::new(0.08, 0.0)
    }

    /// Returns a post AA profile for normal interactive camera movement.
    pub fn interactive() -> Self {
        Self::new(0.040, 0.50)
    }

    /// Returns a balanced post AA profile with lower edge-search cost than inspection mode.
    pub fn balanced() -> Self {
        Self::new(0.026, 0.78)
    }

    /// Returns the high-quality edge resolve profile.
    pub fn high_quality() -> Self {
        Self::new(0.018, 0.94)
    }

    /// Returns the luma/depth/normal edge threshold below which AA is skipped.
    pub fn edge_threshold(self) -> f32 {
        self.edge_threshold
    }

    /// Returns the maximum amount of resolved color blended into edge pixels.
    pub fn blend(self) -> f32 {
        self.blend
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostQualitySettings {
    contrast: f32,
    saturation: f32,
}

impl PostQualitySettings {
    /// Creates bounded renderer-level look multipliers applied after app camera effects.
    pub fn new(contrast: f32, saturation: f32) -> Self {
        Self {
            contrast: finite_clamp(contrast, 0.5, 1.5, 0.94),
            saturation: finite_clamp(saturation, 0.0, 2.0, 1.0),
        }
    }

    /// Returns neutral post settings.
    pub fn natural() -> Self {
        Self::new(1.04, 1.0)
    }

    /// Returns the renderer-side contrast multiplier.
    pub fn contrast(self) -> f32 {
        self.contrast
    }

    /// Returns the renderer-side saturation multiplier.
    pub fn saturation(self) -> f32 {
        self.saturation
    }
}

/// Returns a finite scalar clamped to a protocol-defined range.
fn finite_clamp(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_settings_clamp_invalid_values() {
        let settings = RenderQualitySettings::new(
            SsaoQualitySettings::new(2.0, f32::INFINITY, -1.0, 128),
            AntiAliasingQualitySettings::new(-1.0, 2.0),
            PostQualitySettings::new(f32::INFINITY, -1.0),
        );

        assert_eq!(settings.ssao().sample_count(), 8);
        assert_eq!(settings.anti_aliasing().blend(), 1.0);
        assert!(settings.post().contrast().is_finite());
    }

    #[test]
    fn performance_profile_disables_fullscreen_expensive_effects() {
        let settings = RenderQualitySettings::performance();

        assert_eq!(settings.ssao().intensity(), 0.0);
        assert_eq!(settings.anti_aliasing().blend(), 0.0);
    }
}
