#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderQualitySettings {
    ssao: SsaoQualitySettings,
    ssr: SsrQualitySettings,
    anti_aliasing: AntiAliasingQualitySettings,
    shadow_softening: ShadowSofteningQualitySettings,
    contact_shadow: ContactShadowQualitySettings,
    post: PostQualitySettings,
}

impl RenderQualitySettings {
    /// Creates one full renderer quality profile without relying on defaulted subprofiles.
    fn from_parts(
        ssao: SsaoQualitySettings,
        ssr: SsrQualitySettings,
        anti_aliasing: AntiAliasingQualitySettings,
        shadow_softening: ShadowSofteningQualitySettings,
        contact_shadow: ContactShadowQualitySettings,
        post: PostQualitySettings,
    ) -> Self {
        Self {
            ssao,
            ssr,
            anti_aliasing,
            shadow_softening,
            contact_shadow,
            post,
        }
    }

    /// Creates a complete renderer quality profile for fullscreen post effects.
    pub fn new(
        ssao: SsaoQualitySettings,
        anti_aliasing: AntiAliasingQualitySettings,
        post: PostQualitySettings,
    ) -> Self {
        Self::new_with_ssr(ssao, SsrQualitySettings::balanced(), anti_aliasing, post)
    }

    /// Creates a complete renderer quality profile including screen-space reflections.
    pub fn new_with_ssr(
        ssao: SsaoQualitySettings,
        ssr: SsrQualitySettings,
        anti_aliasing: AntiAliasingQualitySettings,
        post: PostQualitySettings,
    ) -> Self {
        Self::new_with_shadow_softening(
            ssao,
            ssr,
            anti_aliasing,
            ShadowSofteningQualitySettings::balanced(),
            post,
        )
    }

    /// Creates a complete renderer quality profile including post-process shadow cleanup.
    pub fn new_with_shadow_softening(
        ssao: SsaoQualitySettings,
        ssr: SsrQualitySettings,
        anti_aliasing: AntiAliasingQualitySettings,
        shadow_softening: ShadowSofteningQualitySettings,
        post: PostQualitySettings,
    ) -> Self {
        Self::from_parts(
            ssao,
            ssr,
            anti_aliasing,
            shadow_softening,
            ContactShadowQualitySettings::balanced(),
            post,
        )
    }

    /// Creates a complete renderer quality profile including screen-space contact shadows.
    pub fn new_with_contact_shadow(
        ssao: SsaoQualitySettings,
        ssr: SsrQualitySettings,
        anti_aliasing: AntiAliasingQualitySettings,
        contact_shadow: ContactShadowQualitySettings,
        post: PostQualitySettings,
    ) -> Self {
        Self::from_parts(
            ssao,
            ssr,
            anti_aliasing,
            ShadowSofteningQualitySettings::balanced(),
            contact_shadow,
            post,
        )
    }

    /// Returns the lightest profile intended for editing and camera navigation.
    pub fn performance() -> Self {
        Self::from_parts(
            SsaoQualitySettings::disabled(),
            SsrQualitySettings::disabled(),
            AntiAliasingQualitySettings::disabled(),
            ShadowSofteningQualitySettings::disabled(),
            ContactShadowQualitySettings::disabled(),
            PostQualitySettings::natural(),
        )
    }

    /// Returns the interactive profile used when the app does not override renderer quality.
    pub fn interactive() -> Self {
        Self::from_parts(
            SsaoQualitySettings::interactive(),
            SsrQualitySettings::interactive(),
            AntiAliasingQualitySettings::interactive(),
            ShadowSofteningQualitySettings::interactive(),
            ContactShadowQualitySettings::interactive(),
            PostQualitySettings::natural(),
        )
    }

    /// Returns the balanced profile for scenes that can spend more post-process budget.
    pub fn balanced() -> Self {
        Self::from_parts(
            SsaoQualitySettings::balanced(),
            SsrQualitySettings::balanced(),
            AntiAliasingQualitySettings::balanced(),
            ShadowSofteningQualitySettings::balanced(),
            ContactShadowQualitySettings::balanced(),
            PostQualitySettings::natural(),
        )
    }

    /// Returns the expensive profile used when visual inspection matters more than frame time.
    pub fn high_quality() -> Self {
        Self::from_parts(
            SsaoQualitySettings::high_quality(),
            SsrQualitySettings::high_quality(),
            AntiAliasingQualitySettings::high_quality(),
            ShadowSofteningQualitySettings::high_quality(),
            ContactShadowQualitySettings::high_quality(),
            PostQualitySettings::natural(),
        )
    }

    /// Returns a copy with a different screen-space reflection profile.
    pub fn with_ssr(mut self, ssr: SsrQualitySettings) -> Self {
        self.ssr = ssr;
        self
    }

    /// Returns a copy with a different post-process shadow cleanup profile.
    pub fn with_shadow_softening(
        mut self,
        shadow_softening: ShadowSofteningQualitySettings,
    ) -> Self {
        self.shadow_softening = shadow_softening;
        self
    }

    /// Returns a copy with a different screen-space contact shadow profile.
    pub fn with_contact_shadow(mut self, contact_shadow: ContactShadowQualitySettings) -> Self {
        self.contact_shadow = contact_shadow;
        self
    }

    /// Returns the screen-space ambient occlusion quality applied by the post pass.
    pub fn ssao(self) -> SsaoQualitySettings {
        self.ssao
    }

    /// Returns the screen-space reflection quality applied by the post pass.
    pub fn ssr(self) -> SsrQualitySettings {
        self.ssr
    }

    /// Returns the post-pass antialiasing quality applied before tone mapping.
    pub fn anti_aliasing(self) -> AntiAliasingQualitySettings {
        self.anti_aliasing
    }

    /// Returns the post-process shadow cleanup applied before tone mapping.
    pub fn shadow_softening(self) -> ShadowSofteningQualitySettings {
        self.shadow_softening
    }

    /// Returns the screen-space contact shadow quality applied by the post pass.
    pub fn contact_shadow(self) -> ContactShadowQualitySettings {
        self.contact_shadow
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
pub struct ShadowSofteningQualitySettings {
    intensity: f32,
    radius_pixels: f32,
    depth_sensitivity: f32,
    max_luma_delta: f32,
}

impl ShadowSofteningQualitySettings {
    /// Creates bounded controls for post-process cleanup of noisy soft shadow edges.
    ///
    /// The pass upscales low-frequency shadow edges in screen space. It only adjusts
    /// local luminance and uses depth/normal weights, so it can hide lower-resolution
    /// shadow-map stair steps without blurring object silhouettes as aggressively as
    /// a full color blur.
    pub fn new(
        intensity: f32,
        radius_pixels: f32,
        depth_sensitivity: f32,
        max_luma_delta: f32,
    ) -> Self {
        Self {
            intensity: finite_clamp(intensity, 0.0, 1.0, 0.0),
            radius_pixels: finite_clamp(radius_pixels, 0.5, 12.0, 4.5),
            depth_sensitivity: finite_clamp(depth_sensitivity, 0.1, 8.0, 1.7),
            max_luma_delta: finite_clamp(max_luma_delta, 0.005, 0.25, 0.075),
        }
    }

    /// Disables shadow cleanup so the post shader can skip the extra taps.
    pub fn disabled() -> Self {
        Self::new(0.0, 0.75, 1.8, 0.040)
    }

    /// Returns a light cleanup profile for normal camera movement.
    pub fn interactive() -> Self {
        Self::new(0.28, 2.50, 3.6, 0.070)
    }

    /// Returns the default cleanup profile for balanced soft-shadow quality.
    pub fn balanced() -> Self {
        Self::new(0.66, 5.75, 2.45, 0.160)
    }

    /// Returns the stronger cleanup profile for visual inspection.
    pub fn high_quality() -> Self {
        Self::new(0.92, 10.50, 1.85, 0.240)
    }

    /// Returns the final blend strength of the shadow cleanup.
    pub fn intensity(self) -> f32 {
        self.intensity
    }

    /// Returns the filter radius in screen pixels.
    pub fn radius_pixels(self) -> f32 {
        self.radius_pixels
    }

    /// Returns how strongly depth differences reject neighboring taps.
    pub fn depth_sensitivity(self) -> f32 {
        self.depth_sensitivity
    }

    /// Returns the maximum local luma shift applied by the cleanup pass.
    pub fn max_luma_delta(self) -> f32 {
        self.max_luma_delta
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactShadowQualitySettings {
    intensity: f32,
    max_distance: f32,
    thickness: f32,
    sample_count: u32,
}

impl ContactShadowQualitySettings {
    /// Creates bounded controls for the post-pass screen-space contact shadow ray march.
    ///
    /// `max_distance` is measured in view-space world units, `thickness` is the
    /// accepted depth thickness for a hit, and `sample_count` bounds fullscreen cost.
    pub fn new(intensity: f32, max_distance: f32, thickness: f32, sample_count: u32) -> Self {
        Self {
            intensity: finite_clamp(intensity, 0.0, 1.0, 0.0),
            max_distance: finite_clamp(max_distance, 0.05, 3.0, 0.85),
            thickness: finite_clamp(thickness, 0.008, 0.35, 0.070),
            sample_count: sample_count.clamp(1, 24),
        }
    }

    /// Disables contact shadows so the post shader skips the ray march.
    pub fn disabled() -> Self {
        Self::new(0.0, 0.30, 0.070, 1)
    }

    /// Returns a short, low-cost contact shadow profile for camera movement.
    pub fn interactive() -> Self {
        Self::new(0.14, 0.32, 0.070, 6)
    }

    /// Returns the default contact shadow profile for practical scene grounding.
    pub fn balanced() -> Self {
        Self::new(0.24, 0.58, 0.060, 14)
    }

    /// Returns the inspection profile with longer rays and denser sampling.
    pub fn high_quality() -> Self {
        Self::new(0.34, 0.92, 0.052, 24)
    }

    /// Returns the final darkening strength.
    pub fn intensity(self) -> f32 {
        self.intensity
    }

    /// Returns the maximum screen-space shadow ray distance in view-space units.
    pub fn max_distance(self) -> f32 {
        self.max_distance
    }

    /// Returns the accepted view-depth thickness for a contact hit.
    pub fn thickness(self) -> f32 {
        self.thickness
    }

    /// Returns the maximum number of samples evaluated per shaded pixel.
    pub fn sample_count(self) -> u32 {
        self.sample_count
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrQualitySettings {
    intensity: f32,
    max_steps: u32,
    max_distance: f32,
    thickness: f32,
}

impl SsrQualitySettings {
    /// Creates bounded SSR controls for the post-pass DDA tracer.
    ///
    /// `intensity` is the final blend cap, `max_steps` bounds DDA work,
    /// `max_distance` limits view-space ray length, and `thickness` controls
    /// depth-hit tolerance.
    pub fn new(intensity: f32, max_steps: u32, max_distance: f32, thickness: f32) -> Self {
        Self {
            intensity: finite_clamp(intensity, 0.0, 1.0, 0.0),
            max_steps: max_steps.clamp(1, 64),
            max_distance: finite_clamp(max_distance, 0.25, 96.0, 42.0),
            thickness: finite_clamp(thickness, 0.01, 1.0, 0.14),
        }
    }

    /// Disables SSR so the post shader can skip the reflection ray march.
    pub fn disabled() -> Self {
        Self::new(0.0, 1, 8.0, 0.12)
    }

    /// Returns a visible SSR profile that stays responsive during camera movement.
    pub fn interactive() -> Self {
        Self::new(0.34, 16, 22.0, 0.17)
    }

    /// Returns the default cinematic SSR profile used by normal windowed rendering.
    pub fn balanced() -> Self {
        Self::new(0.72, 40, 56.0, 0.13)
    }

    /// Returns the inspection SSR profile with longer rays and cleaner hit refinement.
    pub fn high_quality() -> Self {
        Self::new(0.95, 64, 90.0, 0.11)
    }

    /// Returns the strongest built-in SSR profile for still screenshots and visual checks.
    pub fn ultra() -> Self {
        Self::new(1.0, 64, 96.0, 0.10)
    }

    /// Returns an intentionally exaggerated SSR profile for diagnosing reflective materials.
    pub fn debug_strong() -> Self {
        Self::new(1.0, 64, 96.0, 0.22)
    }

    /// Returns the final reflection blend cap.
    pub fn intensity(self) -> f32 {
        self.intensity
    }

    /// Returns the maximum number of DDA steps evaluated per reflective pixel.
    pub fn max_steps(self) -> u32 {
        self.max_steps
    }

    /// Returns the maximum view-space ray length.
    pub fn max_distance(self) -> f32 {
        self.max_distance
    }

    /// Returns the base depth tolerance used when matching ray depth to scene depth.
    pub fn thickness(self) -> f32 {
        self.thickness
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
        Self::new(0.20, 0.52, 0.032, 2)
    }

    /// Returns an SSAO profile that adds contact depth without crushing shaded surfaces.
    pub fn balanced() -> Self {
        Self::new(0.42, 0.70, 0.030, 4)
    }

    /// Returns the old inspection-quality SSAO profile.
    pub fn high_quality() -> Self {
        Self::new(0.58, 0.85, 0.026, 8)
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
        Self::new(0.034, 0.45)
    }

    /// Returns a balanced post AA profile with lower edge-search cost than inspection mode.
    pub fn balanced() -> Self {
        Self::new(0.018, 0.78)
    }

    /// Returns the high-quality edge resolve profile.
    pub fn high_quality() -> Self {
        Self::new(0.010, 0.98)
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
        assert!(settings.ssr().max_distance().is_finite());
        assert_eq!(settings.anti_aliasing().blend(), 1.0);
        assert!(settings.shadow_softening().radius_pixels().is_finite());
        assert!(settings.contact_shadow().max_distance().is_finite());
        assert!(settings.post().contrast().is_finite());
    }

    #[test]
    fn performance_profile_disables_fullscreen_expensive_effects() {
        let settings = RenderQualitySettings::performance();

        assert_eq!(settings.ssao().intensity(), 0.0);
        assert_eq!(settings.ssr().intensity(), 0.0);
        assert_eq!(settings.anti_aliasing().blend(), 0.0);
        assert_eq!(settings.shadow_softening().intensity(), 0.0);
        assert_eq!(settings.contact_shadow().intensity(), 0.0);
    }

    #[test]
    fn ssr_settings_clamp_shader_work() {
        let settings = SsrQualitySettings::new(5.0, 128, f32::INFINITY, -1.0);

        assert_eq!(settings.intensity(), 1.0);
        assert_eq!(settings.max_steps(), 64);
        assert!(settings.max_distance().is_finite());
        assert_eq!(settings.thickness(), 0.01);
    }

    #[test]
    fn shadow_softening_settings_bound_post_work() {
        let settings = ShadowSofteningQualitySettings::new(5.0, 100.0, f32::INFINITY, -1.0);

        assert_eq!(settings.intensity(), 1.0);
        assert_eq!(settings.radius_pixels(), 12.0);
        assert!(settings.depth_sensitivity().is_finite());
        assert_eq!(settings.max_luma_delta(), 0.005);
    }

    #[test]
    fn contact_shadow_settings_bound_post_work() {
        let settings = ContactShadowQualitySettings::new(5.0, f32::INFINITY, -1.0, 128);

        assert_eq!(settings.intensity(), 1.0);
        assert!(settings.max_distance().is_finite());
        assert_eq!(settings.thickness(), 0.008);
        assert_eq!(settings.sample_count(), 24);
    }

    #[test]
    fn default_visual_profiles_enable_contact_shadows() {
        assert!(
            RenderQualitySettings::interactive()
                .contact_shadow()
                .intensity()
                > 0.0
        );
        assert!(
            RenderQualitySettings::balanced()
                .contact_shadow()
                .intensity()
                > 0.0
        );
        assert!(
            RenderQualitySettings::high_quality()
                .contact_shadow()
                .intensity()
                > 0.0
        );
    }

    #[test]
    fn visual_quality_profiles_have_clear_cost_steps() {
        let interactive = RenderQualitySettings::interactive();
        let balanced = RenderQualitySettings::balanced();
        let high = RenderQualitySettings::high_quality();

        assert!(interactive.ssr().intensity() < balanced.ssr().intensity());
        assert!(balanced.ssr().intensity() < high.ssr().intensity());
        assert!(interactive.ssr().max_steps() < balanced.ssr().max_steps());
        assert!(balanced.ssr().max_steps() < high.ssr().max_steps());

        assert!(interactive.ssao().intensity() < balanced.ssao().intensity());
        assert!(balanced.ssao().intensity() < high.ssao().intensity());
        assert!(interactive.ssao().sample_count() < balanced.ssao().sample_count());
        assert!(balanced.ssao().sample_count() < high.ssao().sample_count());

        assert!(interactive.anti_aliasing().blend() < balanced.anti_aliasing().blend());
        assert!(balanced.anti_aliasing().blend() < high.anti_aliasing().blend());

        assert!(interactive.contact_shadow().intensity() < balanced.contact_shadow().intensity());
        assert!(balanced.contact_shadow().intensity() < high.contact_shadow().intensity());
        assert!(
            interactive.contact_shadow().sample_count() < balanced.contact_shadow().sample_count()
        );
        assert!(balanced.contact_shadow().sample_count() < high.contact_shadow().sample_count());
    }
}
