use valkan_test::{
    app::App,
    renderer::{
        Camera, DirectionalLight, ModelId, PipelineId, RenderModel, RenderScene, Renderer,
        SceneContext, SceneController, SceneKey, SceneMessage, Transform,
    },
};
use winit::event_loop::EventLoop;

const MODEL_CANDIDATES: [&str; 3] = ["assets/model.glb", "assets/model.gltf", "assets/model.obj"];
const MODEL_PATH_ENV: &str = "MODEL_PATH";

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

#[derive(Debug, Clone)]
struct FreeCamera {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    moving_forward: bool,
    moving_back: bool,
    moving_left: bool,
    moving_right: bool,
    moving_down: bool,
    moving_up: bool,
    turning_up: bool,
    turning_down: bool,
    turning_left: bool,
    turning_right: bool,
}

impl Default for FreeCamera {
    fn default() -> Self {
        Self {
            position: [0.0, 1.8, 5.0],
            yaw: std::f32::consts::PI,
            pitch: -0.18,
            moving_forward: false,
            moving_back: false,
            moving_left: false,
            moving_right: false,
            moving_down: false,
            moving_up: false,
            turning_up: false,
            turning_down: false,
            turning_left: false,
            turning_right: false,
        }
    }
}

impl FreeCamera {
    fn set_key(&mut self, key: SceneKey, pressed: bool) {
        match key {
            SceneKey::KeyW => self.moving_forward = pressed,
            SceneKey::KeyS => self.moving_back = pressed,
            SceneKey::KeyA => self.moving_left = pressed,
            SceneKey::KeyD => self.moving_right = pressed,
            SceneKey::KeyQ => self.moving_down = pressed,
            SceneKey::KeyE => self.moving_up = pressed,
            SceneKey::ArrowUp => self.turning_up = pressed,
            SceneKey::ArrowDown => self.turning_down = pressed,
            SceneKey::ArrowLeft => self.turning_left = pressed,
            SceneKey::ArrowRight => self.turning_right = pressed,
            _ => {}
        }
    }

    fn update(&mut self, delta_time: f32) {
        let move_speed = 3.2;
        let turn_speed = 1.8;

        if self.turning_left {
            self.yaw += delta_time * turn_speed;
        }
        if self.turning_right {
            self.yaw -= delta_time * turn_speed;
        }
        if self.turning_up {
            self.pitch += delta_time * turn_speed;
        }
        if self.turning_down {
            self.pitch -= delta_time * turn_speed;
        }
        self.pitch = self.pitch.clamp(-1.35, 1.35);

        let forward = self.forward();
        let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
        let mut velocity = [0.0; 3];

        if self.moving_forward {
            add_scaled(&mut velocity, forward, 1.0);
        }
        if self.moving_back {
            add_scaled(&mut velocity, forward, -1.0);
        }
        if self.moving_right {
            add_scaled(&mut velocity, right, 1.0);
        }
        if self.moving_left {
            add_scaled(&mut velocity, right, -1.0);
        }
        if self.moving_up {
            velocity[1] += 1.0;
        }
        if self.moving_down {
            velocity[1] -= 1.0;
        }

        let velocity = normalize_or_zero(velocity);
        add_scaled(&mut self.position, velocity, delta_time * move_speed);
    }

    fn camera(&self) -> Camera {
        let forward = self.forward();

        Camera {
            eye: self.position,
            target: [
                self.position[0] + forward[0],
                self.position[1] + forward[1],
                self.position[2] + forward[2],
            ],
            ..Camera::default()
        }
    }

    fn forward(&self) -> [f32; 3] {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();

        [yaw_sin * pitch_cos, pitch_sin, yaw_cos * pitch_cos]
    }
}

impl SceneController for MainScene {
    fn on_renderer_ready(&mut self, renderer: &mut Renderer) {
        let model_paths = model_paths();
        self.model = load_first_model(renderer, &model_paths).unwrap_or(ModelId::CUBE);
    }

    fn on_message(&mut self, message: SceneMessage) {
        let SceneMessage::Keyboard { key, pressed } = message else {
            return;
        };

        match key {
            SceneKey::Space if pressed => self.camera = FreeCamera::default(),
            key => self.camera.set_key(key, pressed),
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
            objects: Vec::new(),
            models: vec![RenderModel {
                model: self.model,
                pipeline: PipelineId::LIT_MESH,
                transform: Transform::default(),
            }],
        }
    }
}

fn model_paths() -> Vec<String> {
    let mut paths = Vec::new();

    if let Ok(path) = std::env::var(MODEL_PATH_ENV) {
        if !path.trim().is_empty() {
            paths.push(path);
        }
    }

    paths.extend(MODEL_CANDIDATES.iter().map(|path| (*path).to_string()));
    paths
}

fn load_first_model(renderer: &mut Renderer, paths: &[String]) -> Option<ModelId> {
    for path in paths {
        match renderer.load_model(path.as_str()) {
            Ok(model) => {
                log::info!("loaded model: {path}");
                return Some(model);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => log::warn!("failed to load model '{path}': {err}"),
        }
    }

    None
}

fn add_scaled(target: &mut [f32; 3], value: [f32; 3], scale: f32) {
    target[0] += value[0] * scale;
    target[1] += value[1] * scale;
    target[2] += value[2] * scale;
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    normalize_or_zero(v)
}

fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

    if len <= f32::EPSILON {
        return [0.0; 3];
    }

    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
