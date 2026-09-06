/// Continuous renderer controls plus explicit feature switches.
///
/// The scalar fields remain available for renderer-internal tuning. Use [`Self::with_features`]
/// when a command only needs to turn a feature on or off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderQualitySettings {
    ssao: SsaoQualitySettings,
    ssr: SsrQualitySettings,
    anti_aliasing: AntiAliasingQualitySettings,
    stable_csm_pcss: StableCsmPcssQualitySettings,
    bloom: BloomQualitySettings,
    fog: VolumetricFogQualitySettings,
    post: PostQualitySettings,
    features: RenderFeatureToggles,
}

/// Explicit ON/OFF switches for renderer features.
///
/// This is intentionally separate from [`RenderQualitySettings`]' continuous controls.  A
/// feature command can therefore enable or disable a pass without silently changing its sample
/// budget, radius, or visual strength.  New switches can be added here without changing the
/// numeric quality contract used by the renderer internals.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderFeatureToggles {
    ssao: bool,
    ssr: bool,
    anti_aliasing: bool,
    bloom: bool,
    god_rays: bool,
    volumetric_fog: bool,
}

impl RenderFeatureToggles {
    pub const SSAO_BIT: u32 = 1 << 0;
    pub const SSR_BIT: u32 = 1 << 1;
    pub const ANTI_ALIASING_BIT: u32 = 1 << 2;
    pub const BLOOM_BIT: u32 = 1 << 3;
    pub const GOD_RAYS_BIT: u32 = 1 << 4;
    pub const VOLUMETRIC_FOG_BIT: u32 = 1 << 5;

    /// Returns a feature set with every optional effect disabled.
    pub const fn disabled() -> Self {
        Self {
            ssao: false,
            ssr: false,
            anti_aliasing: false,
            bloom: false,
            god_rays: false,
            volumetric_fog: false,
        }
    }

    /// Key `2`: enable the inexpensive spatial effects needed for navigation.
    pub const fn spatial() -> Self {
        Self {
            anti_aliasing: true,
            ssao: true,
            ..Self::disabled()
        }
    }

    /// Key `3`: add screen-space reflections and the legacy post God Ray path.
    pub const fn post() -> Self {
        Self {
            ssr: true,
            bloom: true,
            god_rays: true,
            ..Self::spatial()
        }
    }

    /// Key `4`: enable every currently implemented optional feature.
    pub const fn full() -> Self {
        Self {
            volumetric_fog: true,
            ..Self::post()
        }
    }

    /// Compatibility names for callers that still refer to the four keyboard levels.
    pub const fn performance() -> Self {
        Self::disabled()
    }

    pub const fn interactive() -> Self {
        Self::spatial()
    }

    pub const fn balanced() -> Self {
        Self::post()
    }

    pub const fn high_quality() -> Self {
        Self::full()
    }

    /// Creates a feature set from a protocol bit mask. Unknown bits are ignored for forward
    /// compatibility with newer clients.
    pub const fn from_bits(bits: u32) -> Self {
        Self {
            ssao: bits & Self::SSAO_BIT != 0,
            ssr: bits & Self::SSR_BIT != 0,
            anti_aliasing: bits & Self::ANTI_ALIASING_BIT != 0,
            bloom: bits & Self::BLOOM_BIT != 0,
            god_rays: bits & Self::GOD_RAYS_BIT != 0,
            volumetric_fog: bits & Self::VOLUMETRIC_FOG_BIT != 0,
        }
    }

    /// Returns the compact protocol representation used by post push constants and traces.
    pub const fn bits(self) -> u32 {
        (if self.ssao { Self::SSAO_BIT } else { 0 })
            | (if self.ssr { Self::SSR_BIT } else { 0 })
            | (if self.anti_aliasing {
                Self::ANTI_ALIASING_BIT
            } else {
                0
            })
            | (if self.bloom { Self::BLOOM_BIT } else { 0 })
            | (if self.god_rays { Self::GOD_RAYS_BIT } else { 0 })
            | (if self.volumetric_fog {
                Self::VOLUMETRIC_FOG_BIT
            } else {
                0
            })
    }

    pub const fn ssao_enabled(self) -> bool {
        self.ssao
    }

    pub const fn ssr_enabled(self) -> bool {
        self.ssr
    }

    pub const fn anti_aliasing_enabled(self) -> bool {
        self.anti_aliasing
    }

    pub const fn bloom_enabled(self) -> bool {
        self.bloom
    }

    pub const fn god_rays_enabled(self) -> bool {
        self.god_rays
    }

    pub const fn volumetric_fog_enabled(self) -> bool {
        self.volumetric_fog
    }

    pub const fn with_ssao(mut self, enabled: bool) -> Self {
        self.ssao = enabled;
        self
    }

    pub const fn with_ssr(mut self, enabled: bool) -> Self {
        self.ssr = enabled;
        self
    }

    pub const fn with_anti_aliasing(mut self, enabled: bool) -> Self {
        self.anti_aliasing = enabled;
        self
    }

    pub const fn with_bloom(mut self, enabled: bool) -> Self {
        self.bloom = enabled;
        self
    }

    pub const fn with_god_rays(mut self, enabled: bool) -> Self {
        self.god_rays = enabled;
        self
    }

    pub const fn with_volumetric_fog(mut self, enabled: bool) -> Self {
        self.volumetric_fog = enabled;
        self
    }
}

impl RenderQualitySettings {
    /// Creates one full renderer quality profile without relying on defaulted subprofiles.
    fn from_parts(
        ssao: SsaoQualitySettings,
        ssr: SsrQualitySettings,
        anti_aliasing: AntiAliasingQualitySettings,
        stable_csm_pcss: StableCsmPcssQualitySettings,
        bloom: BloomQualitySettings,
        fog: VolumetricFogQualitySettings,
        post: PostQualitySettings,
        features: RenderFeatureToggles,
    ) -> Self {
        let mut settings = Self {
            ssao,
            ssr,
            anti_aliasing,
            stable_csm_pcss,
            bloom,
            fog,
            post,
            features,
        };
        settings.sync_feature_routes();
        settings
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
        let features = RenderFeatureToggles::post()
            .with_ssao(ssao.intensity() > 0.0)
            .with_ssr(ssr.intensity() > 0.0)
            .with_anti_aliasing(anti_aliasing.blend() > 0.0);
        Self::from_parts(
            ssao,
            ssr,
            anti_aliasing,
            StableCsmPcssQualitySettings::balanced(),
            BloomQualitySettings::balanced(),
            VolumetricFogQualitySettings::high_quality().with_enabled(false),
            post,
            features,
        )
    }

    /// Returns the low-cost continuous profile used by keyboard preset `1` and renderer tuning.
    pub fn performance() -> Self {
        Self::from_parts(
            SsaoQualitySettings::disabled(),
            SsrQualitySettings::disabled(),
            AntiAliasingQualitySettings::disabled(),
            StableCsmPcssQualitySettings::performance(),
            BloomQualitySettings::disabled(),
            VolumetricFogQualitySettings::high_quality().with_enabled(false),
            PostQualitySettings::natural(),
            RenderFeatureToggles::disabled(),
        )
    }

    /// Returns the interactive continuous profile used by keyboard preset `2`.
    pub fn interactive() -> Self {
        Self::from_parts(
            SsaoQualitySettings::interactive(),
            SsrQualitySettings::interactive(),
            AntiAliasingQualitySettings::interactive(),
            StableCsmPcssQualitySettings::interactive(),
            BloomQualitySettings::interactive(),
            VolumetricFogQualitySettings::high_quality().with_enabled(false),
            PostQualitySettings::natural(),
            RenderFeatureToggles::post(),
        )
    }

    /// Returns the balanced continuous profile used by keyboard preset `3`.
    pub fn balanced() -> Self {
        Self::from_parts(
            SsaoQualitySettings::balanced(),
            SsrQualitySettings::balanced(),
            AntiAliasingQualitySettings::balanced(),
            StableCsmPcssQualitySettings::balanced(),
            BloomQualitySettings::balanced(),
            VolumetricFogQualitySettings::high_quality().with_enabled(false),
            PostQualitySettings::natural(),
            RenderFeatureToggles::post(),
        )
    }

    /// Returns the high-cost continuous profile used by keyboard preset `4`.
    pub fn high_quality() -> Self {
        Self::from_parts(
            SsaoQualitySettings::high_quality(),
            SsrQualitySettings::high_quality(),
            AntiAliasingQualitySettings::high_quality(),
            StableCsmPcssQualitySettings::high_quality(),
            BloomQualitySettings::high_quality(),
            VolumetricFogQualitySettings::high_quality(),
            PostQualitySettings::natural(),
            RenderFeatureToggles::full(),
        )
    }

    /// Returns a copy with a different screen-space reflection profile.
    pub fn with_ssr(mut self, ssr: SsrQualitySettings) -> Self {
        self.ssr = ssr;
        self
    }

    /// Returns a copy with different continuous bloom controls. Feature ON/OFF state is
    /// intentionally unchanged; use [`Self::with_features`] for that.
    pub fn with_bloom(mut self, bloom: BloomQualitySettings) -> Self {
        self.bloom = bloom;
        self.sync_feature_routes();
        self
    }

    /// Returns a copy with different continuous volumetric-fog controls. Feature ON/OFF state is
    /// intentionally unchanged; use [`Self::with_features`] for that.
    pub fn with_fog(mut self, fog: VolumetricFogQualitySettings) -> Self {
        self.fog = fog;
        self.sync_feature_routes();
        self
    }

    /// Returns a copy with explicit feature ON/OFF switches while preserving every continuous
    /// quality value in this settings object.
    pub fn with_features(mut self, features: RenderFeatureToggles) -> Self {
        self.features = features;
        self.sync_feature_routes();
        self
    }

    /// Returns the feature ON/OFF switches carried by this renderer configuration.
    pub fn features(self) -> RenderFeatureToggles {
        self.features
    }

    /// Keeps legacy route fields as derived values for shader/resource compatibility.  The
    /// feature mask remains the single owner of user-visible ON/OFF state.
    fn sync_feature_routes(&mut self) {
        let volumetric = self.features.volumetric_fog_enabled();
        self.bloom = self.bloom.with_volumetric_god_rays(volumetric);
        self.fog = self.fog.with_enabled(volumetric);
    }

    /// Returns the screen-space ambient occlusion quality applied by the post pass.
    pub fn ssao(self) -> SsaoQualitySettings {
        self.ssao
    }

    /// Returns the screen-space reflection quality applied by the post pass.
    pub fn ssr(self) -> SsrQualitySettings {
        self.ssr
    }

    /// Returns the final spatial SMAA quality applied to the complete post result.
    pub fn anti_aliasing(self) -> AntiAliasingQualitySettings {
        self.anti_aliasing
    }

    /// Returns the spatial Stable CSM + PCSS directional-shadow policy.
    pub fn stable_csm_pcss(self) -> StableCsmPcssQualitySettings {
        self.stable_csm_pcss
    }

    /// Returns a copy with a different Stable CSM + PCSS shadow policy.
    pub fn with_stable_csm_pcss(mut self, stable_csm_pcss: StableCsmPcssQualitySettings) -> Self {
        self.stable_csm_pcss = stable_csm_pcss;
        self.sync_feature_routes();
        self
    }

    /// Returns the bloom quality applied by the post pass.
    pub fn bloom(self) -> BloomQualitySettings {
        self.bloom
    }

    /// Returns the volumetric medium used by the quality God Ray path.
    pub fn fog(self) -> VolumetricFogQualitySettings {
        self.fog
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

/// Spatial quality controls for the Stable CSM + PCSS directional shadow path.
///
/// These values describe the fixed four-layer map resolution and spatial work performed by the
/// current frame. Increasing blocker/filter counts improves penumbra stability; temporal reuse is
/// controlled separately by the scene lighting path and is not part of this spatial budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StableCsmPcssQualitySettings {
    blocker_search_samples: u32,
    filter_samples: u32,
    light_angular_radius_radians: f32,
    shadow_map_resolution: u32,
    receiver_bias_scale: f32,
    slope_bias_scale: f32,
    normal_offset_scale: f32,
    receiver_plane_bias_scale: f32,
}

impl StableCsmPcssQualitySettings {
    pub const MAX_BLOCKER_SEARCH_SAMPLES: u32 = 16;
    pub const MAX_FILTER_SAMPLES: u32 = 32;
    pub const MAX_LIGHT_ANGULAR_RADIUS_DEGREES: f32 = 5.0;
    pub const MIN_SHADOW_MAP_RESOLUTION: u32 = 512;
    pub const MAX_SHADOW_MAP_RESOLUTION: u32 = 8192;

    pub fn new(
        blocker_search_samples: u32,
        filter_samples: u32,
        light_angular_radius_degrees: f32,
    ) -> Self {
        Self {
            blocker_search_samples: blocker_search_samples
                .clamp(4, Self::MAX_BLOCKER_SEARCH_SAMPLES),
            filter_samples: filter_samples.clamp(4, Self::MAX_FILTER_SAMPLES),
            light_angular_radius_radians: finite_clamp(
                light_angular_radius_degrees,
                0.0,
                Self::MAX_LIGHT_ANGULAR_RADIUS_DEGREES,
                0.4,
            )
            .to_radians(),
            shadow_map_resolution: 4096,
            receiver_bias_scale: 1.0,
            slope_bias_scale: 1.0,
            normal_offset_scale: 1.0,
            receiver_plane_bias_scale: 1.0,
        }
    }

    /// Selects one shared edge length for all four Stable CSM layers.
    pub fn with_shadow_map_resolution(mut self, resolution: u32) -> Self {
        self.shadow_map_resolution = resolution.clamp(
            Self::MIN_SHADOW_MAP_RESOLUTION,
            Self::MAX_SHADOW_MAP_RESOLUTION,
        );
        self
    }

    pub fn with_receiver_bias_scale(mut self, scale: f32) -> Self {
        self.receiver_bias_scale = finite_clamp(scale, 0.25, 8.0, 1.0);
        self
    }

    /// Scales the angle-dependent depth bias used for grazing receivers.
    pub fn with_slope_bias_scale(mut self, scale: f32) -> Self {
        self.slope_bias_scale = finite_clamp(scale, 0.0, 8.0, 1.0);
        self
    }

    /// Scales the world-space receiver displacement along the interpolated normal.
    pub fn with_normal_offset_scale(mut self, scale: f32) -> Self {
        self.normal_offset_scale = finite_clamp(scale, 0.0, 8.0, 1.0);
        self
    }

    /// Scales the receiver-plane depth gradient used for PCSS tap comparisons.
    pub fn with_receiver_plane_bias_scale(mut self, scale: f32) -> Self {
        self.receiver_plane_bias_scale = finite_clamp(scale, 0.0, 8.0, 1.0);
        self
    }

    pub fn performance() -> Self {
        // Temporarily elevated while validating acne versus peter-panning in the visual smoke.
        Self::new(4, 8, 0.27)
            .with_shadow_map_resolution(2048)
            .with_receiver_bias_scale(3.0)
    }

    pub fn interactive() -> Self {
        Self::new(6, 12, 0.32)
            .with_shadow_map_resolution(3072)
            .with_receiver_bias_scale(3.0)
    }

    pub fn balanced() -> Self {
        Self::new(10, 16, 0.40)
            .with_shadow_map_resolution(4096)
            // The 4096² profile needs a larger receiver margin than the editing presets. Keep all
            // geometric terms elevated together so grazing planes do not reintroduce acne when
            // PCSS averages a wide, regular tap pattern.
            .with_receiver_bias_scale(4.0)
            .with_slope_bias_scale(1.5)
            .with_normal_offset_scale(1.5)
            .with_receiver_plane_bias_scale(1.5)
    }

    pub fn high_quality() -> Self {
        Self::new(16, 32, 0.60)
            // Quality-first mode uses the maximum shared map size.  Every cascade receives the
            // same 8192² target; there is still no near-only layer or extra CSM stage.
            .with_shadow_map_resolution(8192)
            .with_receiver_bias_scale(6.0)
            .with_slope_bias_scale(2.5)
            .with_normal_offset_scale(2.5)
            .with_receiver_plane_bias_scale(2.5)
    }

    pub fn blocker_search_samples(self) -> u32 {
        self.blocker_search_samples
    }

    pub fn filter_samples(self) -> u32 {
        self.filter_samples
    }

    pub fn light_angular_radius_radians(self) -> f32 {
        self.light_angular_radius_radians
    }

    pub fn light_angular_radius_degrees(self) -> f32 {
        self.light_angular_radius_radians.to_degrees()
    }

    pub fn shadow_map_resolution(self) -> u32 {
        self.shadow_map_resolution
    }

    pub fn receiver_bias_scale(self) -> f32 {
        self.receiver_bias_scale
    }

    pub fn slope_bias_scale(self) -> f32 {
        self.slope_bias_scale
    }

    pub fn normal_offset_scale(self) -> f32 {
        self.normal_offset_scale
    }

    pub fn receiver_plane_bias_scale(self) -> f32 {
        self.receiver_plane_bias_scale
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

    /// Returns an SSAO profile that adds near-field depth without crushing shaded surfaces.
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

    /// Returns the quality tier that the GTAO shader expands into slices and steps.
    pub fn sample_count(self) -> u32 {
        self.sample_count
    }

    /// Returns the number of opposite-direction horizon slices used by the GTAO shader.
    ///
    /// Existing serialized quality values remain valid while the practical work budget is explicit
    /// in diagnostics.
    pub fn slice_count(self) -> u32 {
        match self.sample_count {
            0..=2 => 2,
            3..=4 => 3,
            _ => 4,
        }
    }

    /// Returns the number of radial samples taken on each side of one GTAO slice.
    pub fn steps_per_slice(self) -> u32 {
        match self.sample_count {
            0..=2 => 2,
            3..=4 => 3,
            _ => 4,
        }
    }

    /// Returns the total number of depth taps evaluated by one GTAO pixel.
    pub fn sample_budget(self) -> u32 {
        self.slice_count() * self.steps_per_slice() * 2
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AntiAliasingQualitySettings {
    edge_threshold: f32,
    blend: f32,
}

impl AntiAliasingQualitySettings {
    /// Creates bounded SMAA controls. `blend` controls the spatial neighborhood contribution and
    /// `edge_threshold` controls the luma/geometry edge detector.
    pub fn new(edge_threshold: f32, blend: f32) -> Self {
        Self {
            edge_threshold: finite_clamp(edge_threshold, 0.004, 0.08, 0.028),
            blend: finite_clamp(blend, 0.0, 1.0, 0.78),
        }
    }

    /// Disables spatial anti-aliasing for the lowest-latency profile.
    pub fn disabled() -> Self {
        Self::new(0.08, 0.0)
    }

    /// Returns a responsive SMAA profile for interactive camera movement.
    pub fn interactive() -> Self {
        Self::new(0.034, 0.45)
    }

    /// Returns the default balanced SMAA blend.
    pub fn balanced() -> Self {
        Self::new(0.018, 0.78)
    }

    /// Returns the strongest SMAA blend.
    pub fn high_quality() -> Self {
        Self::new(0.010, 0.98)
    }

    /// Returns the luma/depth/normal edge threshold below which AA is skipped.
    pub fn edge_threshold(self) -> f32 {
        self.edge_threshold
    }

    /// Returns the spatial SMAA neighborhood blend in the range zero through one.
    pub fn blend(self) -> f32 {
        self.blend
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomQualitySettings {
    intensity: f32,
    threshold: f32,
    radius_pixels: f32,
    god_rays_intensity: f32,
    volumetric_god_rays: bool,
}

impl BloomQualitySettings {
    /// Creates bounded controls for the post-pass bloom and god rays.
    ///
    /// `intensity` controls the broad glow, `threshold` selects HDR highlights,
    /// `radius_pixels` controls how far bloom taps spread, and
    /// `god_rays_intensity` controls volumetric rays from visible light sources.
    pub fn new(
        intensity: f32,
        threshold: f32,
        radius_pixels: f32,
        god_rays_intensity: f32,
    ) -> Self {
        Self {
            intensity: finite_clamp(intensity, 0.0, 2.0, 0.0),
            threshold: finite_clamp(threshold, 0.2, 8.0, 1.1),
            radius_pixels: finite_clamp(radius_pixels, 0.5, 32.0, 8.0),
            god_rays_intensity: finite_clamp(god_rays_intensity, 0.0, 1.0, 0.0),
            volumetric_god_rays: false,
        }
    }

    /// Disables bloom and god rays so the post shader can skip the taps.
    pub fn disabled() -> Self {
        Self::new(0.0, 1.50, 1.0, 0.0)
    }

    /// Returns a subtle bloom profile for normal camera movement.
    pub fn interactive() -> Self {
        Self::new(0.025, 1.55, 6.0, 0.12)
    }

    /// Returns a visible bloom profile for regular rendering.
    pub fn balanced() -> Self {
        Self::new(0.060, 1.38, 13.0, 0.28)
    }

    /// Returns a stronger profile intended for visual inspection.
    pub fn high_quality() -> Self {
        Self::new(0.095, 1.24, 22.0, 0.44).with_volumetric_god_rays(true)
    }

    /// Returns the broad glow contribution.
    pub fn intensity(self) -> f32 {
        self.intensity
    }

    /// Returns the HDR luminance threshold used by bright-pass extraction.
    pub fn threshold(self) -> f32 {
        self.threshold
    }

    /// Returns the post-process sampling radius in screen pixels.
    pub fn radius_pixels(self) -> f32 {
        self.radius_pixels
    }

    /// Returns the volumetric ray contribution. Balanced profiles use the legacy screen-space
    /// approximation; the high-quality profile switches the same intensity to the shadow-aware
    /// integration path.
    pub fn god_rays_intensity(self) -> f32 {
        self.god_rays_intensity
    }

    /// Sets the derived route marker used by low-level callers. For a complete renderer
    /// configuration, prefer [`RenderQualitySettings::with_features`] so the feature owner is
    /// unambiguous.
    pub fn with_volumetric_god_rays(mut self, enabled: bool) -> Self {
        self.volumetric_god_rays = enabled;
        self
    }

    /// Returns whether god rays should be integrated through the directional shadow map.
    pub fn volumetric_god_rays(self) -> bool {
        self.volumetric_god_rays
    }
}

/// Controls the world-space medium used by the quality volumetric pass.
///
/// Fog is intentionally separate from the legacy screen-space God Ray intensity. The medium is
/// integrated along every camera ray; the directional light only adds anisotropic in-scattering
/// inside that medium, so fog remains visible when the sun is outside the camera frustum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumetricFogQualitySettings {
    enabled: bool,
    density: f32,
    height_falloff: f32,
    height: f32,
    max_distance: f32,
}

impl VolumetricFogQualitySettings {
    /// Creates a bounded exponential-height fog profile.
    pub fn new(density: f32, height_falloff: f32, height: f32, max_distance: f32) -> Self {
        Self {
            enabled: density.is_finite() && density > 0.0,
            density: finite_clamp(density, 0.0, 0.08, 0.0035),
            height_falloff: finite_clamp(height_falloff, 0.0, 2.0, 0.018),
            height: finite_clamp(height, -1000.0, 1000.0, 0.0),
            max_distance: finite_clamp(max_distance, 1.0, 512.0, 160.0),
        }
    }

    /// Disables volumetric fog and keeps the legacy God Ray path untouched.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            density: 0.0,
            height_falloff: 0.018,
            height: 0.0,
            max_distance: 160.0,
        }
    }

    /// Returns the quality fog profile used for still-image inspection.
    pub fn high_quality() -> Self {
        Self::new(0.0035, 0.018, 0.0, 160.0)
    }

    /// Enables or disables the medium without changing its tuned physical parameters. This is a
    /// derived route marker for low-level callers; complete renderer configurations should use
    /// [`RenderQualitySettings::with_features`] for the ON/OFF decision.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled && self.density > 0.0;
        self
    }

    pub fn enabled(self) -> bool {
        self.enabled
    }

    /// Extinction coefficient in inverse world units.
    pub fn density(self) -> f32 {
        self.density
    }

    /// Exponential density falloff per world unit above `height`.
    pub fn height_falloff(self) -> f32 {
        self.height_falloff
    }

    /// Reference world-space height of the medium.
    pub fn height(self) -> f32 {
        self.height
    }

    /// Maximum distance marched through the medium for a background pixel.
    pub fn max_distance(self) -> f32 {
        self.max_distance
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
        assert!(
            settings
                .stable_csm_pcss()
                .light_angular_radius_radians()
                .is_finite()
        );
        assert!(settings.bloom().threshold().is_finite());
        assert!(settings.fog().max_distance().is_finite());
        assert!(settings.post().contrast().is_finite());
    }

    #[test]
    fn feature_presets_are_explicit_on_off_sets() {
        let disabled = RenderFeatureToggles::disabled();
        let spatial = RenderFeatureToggles::spatial();
        let post = RenderFeatureToggles::post();
        let full = RenderFeatureToggles::full();

        assert_eq!(disabled.bits(), 0);
        assert!(spatial.anti_aliasing_enabled() && spatial.ssao_enabled());
        assert!(!spatial.ssr_enabled() && !spatial.bloom_enabled());
        assert!(post.ssr_enabled() && post.bloom_enabled() && post.god_rays_enabled());
        assert!(!post.volumetric_fog_enabled());
        assert!(full.volumetric_fog_enabled());
        assert_eq!(RenderFeatureToggles::from_bits(full.bits()), full);
        assert_eq!(
            RenderFeatureToggles::from_bits(full.bits() | (1 << 31)),
            full
        );
    }

    #[test]
    fn feature_switches_preserve_continuous_quality_values() {
        let settings = RenderQualitySettings::high_quality();
        let switched = settings.with_features(RenderFeatureToggles::disabled());

        assert_eq!(switched.ssr(), settings.ssr());
        assert_eq!(switched.ssao(), settings.ssao());
        assert_eq!(switched.anti_aliasing(), settings.anti_aliasing());
        assert_eq!(switched.bloom().intensity(), settings.bloom().intensity());
        assert_eq!(switched.bloom().threshold(), settings.bloom().threshold());
        assert_eq!(
            switched.bloom().radius_pixels(),
            settings.bloom().radius_pixels()
        );
        assert_eq!(
            switched.bloom().god_rays_intensity(),
            settings.bloom().god_rays_intensity()
        );
        assert!(!switched.bloom().volumetric_god_rays());
        assert!(!switched.fog().enabled());
        assert_eq!(switched.features(), RenderFeatureToggles::disabled());
    }

    #[test]
    fn performance_profile_disables_fullscreen_expensive_effects() {
        let settings = RenderQualitySettings::performance();

        assert_eq!(settings.ssao().intensity(), 0.0);
        assert_eq!(settings.ssr().intensity(), 0.0);
        assert_eq!(settings.anti_aliasing().blend(), 0.0);
        assert_eq!(settings.bloom().intensity(), 0.0);
        assert_eq!(settings.bloom().god_rays_intensity(), 0.0);
        assert!(!settings.fog().enabled());
    }

    #[test]
    fn ssao_profiles_expand_to_bounded_gtao_budgets() {
        let interactive = RenderQualitySettings::interactive().ssao();
        let balanced = RenderQualitySettings::balanced().ssao();
        let high = RenderQualitySettings::high_quality().ssao();

        assert_eq!(interactive.sample_count(), 2);
        assert_eq!(interactive.slice_count(), 2);
        assert_eq!(interactive.steps_per_slice(), 2);
        assert_eq!(interactive.sample_budget(), 8);
        assert_eq!(balanced.slice_count(), 3);
        assert_eq!(balanced.steps_per_slice(), 3);
        assert_eq!(balanced.sample_budget(), 18);
        assert_eq!(high.slice_count(), 4);
        assert_eq!(high.steps_per_slice(), 4);
        assert_eq!(high.sample_budget(), 32);
        assert!(interactive.sample_budget() < balanced.sample_budget());
        assert!(balanced.sample_budget() < high.sample_budget());
    }

    #[test]
    fn stable_csm_pcss_settings_bound_current_frame_work() {
        let low = StableCsmPcssQualitySettings::new(0, 0, f32::NAN);
        let high = StableCsmPcssQualitySettings::new(99, 1, 99.0)
            .with_shadow_map_resolution(u32::MAX)
            .with_receiver_bias_scale(f32::INFINITY)
            .with_slope_bias_scale(f32::INFINITY)
            .with_normal_offset_scale(f32::NAN)
            .with_receiver_plane_bias_scale(f32::INFINITY);

        assert_eq!(low.blocker_search_samples(), 4);
        assert_eq!(low.filter_samples(), 4);
        assert!((low.light_angular_radius_degrees() - 0.4).abs() < 1.0e-5);
        assert_eq!(high.blocker_search_samples(), 16);
        assert_eq!(high.filter_samples(), 4);
        assert!((high.light_angular_radius_degrees() - 5.0).abs() < 1.0e-5);
        assert_eq!(high.shadow_map_resolution(), 8192);
        assert_eq!(high.receiver_bias_scale(), 1.0);
        assert_eq!(high.slope_bias_scale(), 1.0);
        assert_eq!(high.normal_offset_scale(), 1.0);
        assert_eq!(high.receiver_plane_bias_scale(), 1.0);
    }

    #[test]
    fn stable_csm_pcss_profiles_scale_blocker_and_filter_work() {
        let interactive = RenderQualitySettings::interactive().stable_csm_pcss();
        let balanced = RenderQualitySettings::balanced().stable_csm_pcss();
        let high = RenderQualitySettings::high_quality().stable_csm_pcss();

        assert!(interactive.blocker_search_samples() < balanced.blocker_search_samples());
        assert!(balanced.blocker_search_samples() < high.blocker_search_samples());
        assert!(interactive.filter_samples() < balanced.filter_samples());
        assert!(balanced.filter_samples() < high.filter_samples());
        assert!(
            interactive.light_angular_radius_radians() < balanced.light_angular_radius_radians()
        );
        assert!(balanced.light_angular_radius_radians() < high.light_angular_radius_radians());
        assert!(interactive.shadow_map_resolution() < balanced.shadow_map_resolution());
        assert!(balanced.shadow_map_resolution() < high.shadow_map_resolution());
        assert_eq!(high.shadow_map_resolution(), 8192);
        assert!(balanced.receiver_bias_scale() > interactive.receiver_bias_scale());
        assert!(balanced.slope_bias_scale() > interactive.slope_bias_scale());
        assert!(balanced.normal_offset_scale() > interactive.normal_offset_scale());
        assert!(balanced.receiver_plane_bias_scale() > interactive.receiver_plane_bias_scale());
        assert!(high.receiver_bias_scale() > balanced.receiver_bias_scale());
        assert!(high.slope_bias_scale() > balanced.slope_bias_scale());
        assert!(high.normal_offset_scale() > balanced.normal_offset_scale());
        assert!(high.receiver_plane_bias_scale() > balanced.receiver_plane_bias_scale());
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
    fn bloom_settings_bound_post_work() {
        let settings = BloomQualitySettings::new(5.0, f32::INFINITY, -1.0, 5.0);

        assert_eq!(settings.intensity(), 2.0);
        assert!(settings.threshold().is_finite());
        assert_eq!(settings.radius_pixels(), 0.5);
        assert_eq!(settings.god_rays_intensity(), 1.0);
    }

    #[test]
    fn volumetric_fog_settings_bound_medium_work() {
        let settings =
            VolumetricFogQualitySettings::new(f32::INFINITY, -1.0, f32::NAN, f32::INFINITY);

        assert!(!settings.enabled());
        assert!(settings.density().is_finite());
        assert_eq!(settings.height_falloff(), 0.0);
        assert!(settings.height().is_finite());
        assert_eq!(settings.max_distance(), 160.0);
        assert!(RenderQualitySettings::high_quality().fog().enabled());
        assert!(!RenderQualitySettings::balanced().fog().enabled());
    }

    #[test]
    fn default_visual_profiles_enable_bloom() {
        assert!(RenderQualitySettings::interactive().bloom().intensity() > 0.0);
        assert!(RenderQualitySettings::balanced().bloom().intensity() > 0.0);
        assert!(RenderQualitySettings::high_quality().bloom().intensity() > 0.0);
        assert!(
            !RenderQualitySettings::balanced()
                .bloom()
                .volumetric_god_rays()
        );
        assert!(
            RenderQualitySettings::high_quality()
                .bloom()
                .volumetric_god_rays()
        );
        assert!(
            RenderQualitySettings::high_quality()
                .bloom()
                .god_rays_intensity()
                > 0.0
        );
    }

    #[test]
    fn continuous_quality_profiles_have_clear_cost_steps() {
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

        assert!(interactive.bloom().intensity() < balanced.bloom().intensity());
        assert!(balanced.bloom().intensity() < high.bloom().intensity());
        assert!(interactive.bloom().radius_pixels() < balanced.bloom().radius_pixels());
        assert!(balanced.bloom().radius_pixels() < high.bloom().radius_pixels());
        assert!(interactive.bloom().god_rays_intensity() < balanced.bloom().god_rays_intensity());
        assert!(balanced.bloom().god_rays_intensity() < high.bloom().god_rays_intensity());
    }
}
