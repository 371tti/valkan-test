use valkan_test::renderer::{
    DirectionalLight, ModelId, PipelineId, RenderModel, RenderScene, Renderer, SceneContext,
    SceneController, SceneKey, SceneMessage, Transform,
};

use super::{camera::FreeCamera, model_loading::load_scene_model};

pub struct MainScene {
    camera: FreeCamera,
    model: ModelId,
}

impl Default for MainScene {
    fn default() -> Self {
        Self {
            camera: FreeCamera::default(),
            model: ModelId::CUBE,
        }
    }
}

impl SceneController for MainScene {
    fn on_renderer_ready(&mut self, renderer: &mut Renderer) {
        self.model = load_scene_model(renderer);
    }

    fn on_message(&mut self, message: SceneMessage) {
        match message {
            SceneMessage::Keyboard { key, pressed } => match key {
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

        RenderScene {
            camera: self.camera.camera(),
            light: DirectionalLight {
                direction: [-0.35, -0.75, -0.55],
                ambient: [0.11, 0.12, 0.14],
                intensity: 1.55,
                ..DirectionalLight::default()
            },
            reflections: Default::default(),
            objects: Vec::new(),
            models: vec![RenderModel {
                model: self.model,
                pipeline: PipelineId::LIT_MESH,
                transform: Transform::default(),
            }],
        }
    }
}
