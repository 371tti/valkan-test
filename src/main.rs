use valkan_test::{
    app::App,
    renderer::{
        Camera, DirectionalLight, ModelId, PipelineId, RenderModel, RenderScene, Renderer,
        SceneContext, SceneController, SceneKey, SceneMessage, Transform,
    },
};
use winit::event_loop::EventLoop;

fn main() {
    #[cfg(debug_assertions)]
    {
        env_logger::init_from_env(env_logger::Env::default().filter_or("LOG_LEVEL", "trace"));
    }

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(MainScene::default())
        .window_title("Ash Vulkan")
        .window_size(2400, 1600)
        .window_min_size(240, 160)
        .window_transparent(true);

    event_loop.run_app(&mut app).unwrap();
}

struct MainScene {
    angle: f32,
    speed: f32,
    paused: bool,
    light_yaw: f32,
    model: ModelId,
}

impl Default for MainScene {
    fn default() -> Self {
        Self {
            angle: 0.0,
            speed: 1.0,
            paused: false,
            light_yaw: 0.0,
            model: ModelId::CUBE,
        }
    }
}

impl SceneController for MainScene {
    fn on_renderer_ready(&mut self, renderer: &mut Renderer) {
        if let Ok(model) = renderer.load_obj("assets/model.obj") {
            self.model = model;
        }
    }

    fn on_message(&mut self, message: SceneMessage) {
        let SceneMessage::Keyboard { key, pressed } = message else {
            return;
        };

        if !pressed {
            return;
        }

        match key {
            SceneKey::Space => self.paused = !self.paused,
            SceneKey::ArrowUp => self.speed += 0.25,
            SceneKey::ArrowDown => self.speed = (self.speed - 0.25).max(0.0),
            SceneKey::ArrowLeft => self.light_yaw -= 0.15,
            SceneKey::ArrowRight => self.light_yaw += 0.15,
            SceneKey::Other => {}
        }
    }

    fn scene(&mut self, context: SceneContext) -> RenderScene {
        if !self.paused {
            self.angle += context.delta_time * self.speed;
        }

        let (light_sin, light_cos) = self.light_yaw.sin_cos();

        RenderScene {
            camera: Camera::default(),
            light: DirectionalLight {
                direction: [-0.35 * light_cos, -0.75, -0.55 + light_sin * 0.35],
                ..DirectionalLight::default()
            },
            objects: Vec::new(),
            models: vec![RenderModel {
                model: self.model,
                pipeline: PipelineId::LIT_MESH,
                transform: Transform {
                    rotation: [self.angle * 0.57, self.angle, 0.0],
                    ..Transform::default()
                },
            }],
        }
    }
}
