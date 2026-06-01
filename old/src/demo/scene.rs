use valkan_test::renderer::{
    CameraMetering, CameraResponse, DirectionalLight, ModelId, PipelineId, RenderDebugMode,
    RenderModel, RenderScene, Renderer, SceneContext, SceneController, SceneKey, SceneMessage,
    Transform,
};

use super::{camera::FreeCamera, model_loading::load_scene_model};

pub struct MainScene {
    camera: FreeCamera,
    model: Option<ModelId>,
    debug_mode: RenderDebugMode,
    camera_response: AutoCameraResponse,
    shadow_lit_strength: f32,
    light_brighter: bool,
    light_darker: bool,
    exposure_hold: f32,
}

#[derive(Clone, Copy)]
struct AutoCameraResponse {
    exposure: f32,
    white_balance: [f32; 3],
}

const MIN_CAMERA_EXPOSURE: f32 = 0.35;
const MAX_CAMERA_EXPOSURE: f32 = 3.0;
const TARGET_CAMERA_LUMA: f32 = 0.20;
const EXPOSURE_RISE_STOPS_PER_SECOND: f32 = 1.1;
const EXPOSURE_FALL_STOPS_PER_SECOND: f32 = 3.0;
const MIN_WHITE_BALANCE: f32 = 0.55;
const MAX_WHITE_BALANCE: f32 = 1.85;

impl Default for MainScene {
    fn default() -> Self {
        Self {
            camera: FreeCamera::default(),
            model: None,
            debug_mode: RenderDebugMode::Default,
            camera_response: AutoCameraResponse::default(),
            shadow_lit_strength: 1.6,
            light_brighter: false,
            light_darker: false,
            exposure_hold: 0.0,
        }
    }
}

impl SceneController for MainScene {
    fn on_renderer_ready(&mut self, renderer: &mut Renderer) {
        self.model = load_scene_model(renderer);
        if let Some(model) = self.model
            && let Some(bounds) = renderer.model_bounds(model)
        {
            self.camera.frame_sphere(bounds.center, bounds.radius);
            log::info!(
                "framed model: center=({:.2}, {:.2}, {:.2}), radius={:.2}",
                bounds.center[0],
                bounds.center[1],
                bounds.center[2],
                bounds.radius
            );
        }
    }

    fn on_message(&mut self, message: SceneMessage) {
        match message {
            SceneMessage::Keyboard { key, pressed } => match key {
                SceneKey::F12 if pressed => {
                    self.debug_mode = self.debug_mode.next();
                    log::info!("render debug mode: {:?}", self.debug_mode);
                }
                SceneKey::ArrowUp => self.light_brighter = pressed,
                SceneKey::ArrowDown => self.light_darker = pressed,
                SceneKey::Escape if pressed => self.camera.stop(),
                key => self.camera.set_key(key, pressed),
            },
            SceneMessage::MouseMotion { delta } => self.camera.add_mouse_delta(delta),
            SceneMessage::MouseWheel { delta } => self.camera.adjust_speed_multiplier(delta),
            _ => {}
        }
    }

    fn scene(&mut self, context: SceneContext) -> RenderScene {
        self.camera.update(context.delta_time);
        self.update_light(context.delta_time);

        let light = DirectionalLight {
            direction: [0.35, -0.75, 0.55],
            ambient: [0.06, 0.065, 0.075],
            intensity: self.shadow_lit_strength,
            ..DirectionalLight::default()
        };

        let camera_response = if matches!(
            self.debug_mode,
            RenderDebugMode::Default | RenderDebugMode::NoTexture
        ) && self.exposure_hold <= 0.0
        {
            self.camera_response
                .update(context.metering, context.delta_time)
        } else {
            self.camera_response.response()
        };

        let models = self
            .model
            .map(|model| {
                vec![RenderModel {
                    model,
                    pipeline: PipelineId::LIT_MESH,
                    transform: Transform::default(),
                }]
            })
            .unwrap_or_default();

        RenderScene {
            camera: self.camera.camera(),
            camera_response,
            light,
            reflections: Default::default(),
            debug_mode: self.debug_mode,
            objects: Vec::new(),
            models,
        }
    }
}

impl Default for AutoCameraResponse {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            white_balance: [1.0; 3],
        }
    }
}

impl AutoCameraResponse {
    fn response(&self) -> CameraResponse {
        CameraResponse::enabled(
            self.exposure
                .clamp(MIN_CAMERA_EXPOSURE, MAX_CAMERA_EXPOSURE),
            clamp_white_balance(self.white_balance),
        )
    }

    fn update(&mut self, metering: CameraMetering, delta_time: f32) -> CameraResponse {
        let delta_time = delta_time.clamp(0.0, 0.25);

        if metering.valid {
            let metered_luma =
                (metering.center_luminance * 0.65 + metering.average_luminance * 0.35).max(0.002);

            let low_light = 1.0 - smoothstep(0.035, 0.22, metered_luma);

            let target_display_luma = smooth_mix(TARGET_CAMERA_LUMA, 0.26, low_light);
            let max_exposure = smooth_mix(2.2, MAX_CAMERA_EXPOSURE, low_light);

            let highlight_guard = (1.0_f32
                - (metering.highlight_fraction - 0.04_f32).max(0.0_f32) * 1.6_f32)
                .clamp(0.60_f32, 1.0_f32);

            let mut target_exposure =
                self.exposure * (target_display_luma / metered_luma) * highlight_guard;

            target_exposure = target_exposure.clamp(MIN_CAMERA_EXPOSURE, max_exposure);

            let exposure_seconds = if target_exposure < self.exposure {
                smooth_mix(0.15, 0.35, low_light)
            } else {
                smooth_mix(0.55, 1.4, low_light)
            };

            let smoothed =
                smooth_exposure(self.exposure, target_exposure, delta_time, exposure_seconds)
                    .clamp(MIN_CAMERA_EXPOSURE, MAX_CAMERA_EXPOSURE);
            self.exposure = limit_exposure_step(self.exposure, smoothed, delta_time);

            if metering.white_balance_confidence > 0.08 {
                let target_wb = white_balance_for_screen(metering, self.white_balance);
                let wb_seconds = 4.0 / metering.white_balance_confidence.clamp(0.25, 1.0);

                for (current, target) in
                    self.white_balance.iter_mut().zip(target_wb.iter().copied())
                {
                    *current = smooth_exposure(*current, target, delta_time, wb_seconds)
                        .clamp(MIN_WHITE_BALANCE, MAX_WHITE_BALANCE);
                }
            }
        }

        self.response()
    }
}

impl MainScene {
    fn update_light(&mut self, delta_time: f32) {
        let multiplier = 2.0_f32.powf(delta_time * 1.6);
        let mut changed = false;

        if self.light_brighter {
            self.shadow_lit_strength *= multiplier;
            changed = true;
        }
        if self.light_darker {
            self.shadow_lit_strength /= multiplier;
            changed = true;
        }

        if changed {
            self.exposure_hold = 0.8;
        } else {
            self.exposure_hold = (self.exposure_hold - delta_time).max(0.0);
        }
    }
}

fn white_balance_for_screen(metering: CameraMetering, current: [f32; 3]) -> [f32; 3] {
    let gray = luminance(metering.average_color).max(0.001);

    std::array::from_fn(|channel| {
        let correction = (gray / metering.average_color[channel].max(0.001)).clamp(0.72, 1.38);
        (current[channel] * correction.powf(0.45)).clamp(MIN_WHITE_BALANCE, MAX_WHITE_BALANCE)
    })
}

fn clamp_white_balance(value: [f32; 3]) -> [f32; 3] {
    value.map(|channel| channel.clamp(MIN_WHITE_BALANCE, MAX_WHITE_BALANCE))
}

fn luminance(color: [f32; 3]) -> f32 {
    color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
}

fn smooth_toward(current: f32, target: f32, delta_time: f32, seconds: f32) -> f32 {
    let weight = 1.0 - (-delta_time / seconds.max(0.001)).exp();

    smooth_mix(current, target, weight)
}

fn smooth_exposure(current: f32, target: f32, delta_time: f32, seconds: f32) -> f32 {
    smooth_toward(
        current.max(0.0001).ln(),
        target.max(0.0001).ln(),
        delta_time,
        seconds,
    )
    .exp()
}

fn limit_exposure_step(current: f32, target: f32, delta_time: f32) -> f32 {
    let current = current
        .clamp(MIN_CAMERA_EXPOSURE, MAX_CAMERA_EXPOSURE)
        .max(0.0001);
    let rise = 2.0_f32.powf(EXPOSURE_RISE_STOPS_PER_SECOND * delta_time);
    let fall = 2.0_f32.powf(EXPOSURE_FALL_STOPS_PER_SECOND * delta_time);

    target
        .clamp(current / fall, current * rise)
        .clamp(MIN_CAMERA_EXPOSURE, MAX_CAMERA_EXPOSURE)
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(0.0001)).clamp(0.0, 1.0);

    t * t * (3.0 - 2.0 * t)
}

fn smooth_mix(a: f32, b: f32, weight: f32) -> f32 {
    a + (b - a) * weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_exposure_recovers_downward_after_bright_metering() {
        let mut response = AutoCameraResponse {
            exposure: 3.0,
            white_balance: [1.0; 3],
        };

        for _ in 0..8 {
            response.update(metering(0.68, 0.18), 1.0 / 30.0);
        }

        assert!(response.exposure < 1.8);
    }

    #[test]
    fn auto_exposure_stays_in_camera_bounds() {
        let mut response = AutoCameraResponse {
            exposure: 1.0,
            white_balance: [0.1, 4.0, 1.0],
        };

        for _ in 0..180 {
            response.update(metering(0.001, 0.0), 1.0 / 60.0);
        }
        assert!(response.exposure <= MAX_CAMERA_EXPOSURE);
        assert!(response.response().white_balance[0] >= MIN_WHITE_BALANCE);
        assert!(response.response().white_balance[1] <= MAX_WHITE_BALANCE);

        for _ in 0..180 {
            response.update(metering(4.0, 0.8), 1.0 / 60.0);
        }
        assert!(response.exposure >= MIN_CAMERA_EXPOSURE);
    }

    fn metering(luma: f32, highlight_fraction: f32) -> CameraMetering {
        CameraMetering {
            valid: true,
            average_luminance: luma,
            center_luminance: luma,
            highlight_fraction,
            average_color: [luma; 3],
            white_balance_confidence: 0.0,
        }
    }
}
