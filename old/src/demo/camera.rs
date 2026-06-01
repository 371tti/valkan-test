use valkan_test::renderer::{Camera, SceneKey};

#[derive(Debug, Clone)]
pub struct FreeCamera {
    position: [f32; 3],
    velocity: [f32; 3],
    speed_multiplier: f32,
    yaw: f32,
    pitch: f32,
    moving_forward: bool,
    moving_back: bool,
    moving_left: bool,
    moving_right: bool,
    moving_down: bool,
    moving_up: bool,
    sprinting: bool,
    mouse_delta: [f32; 2],
}

impl Default for FreeCamera {
    fn default() -> Self {
        Self {
            position: [0.0, 1.8, 5.0],
            velocity: [0.0; 3],
            speed_multiplier: 1.0,
            yaw: std::f32::consts::PI,
            pitch: -0.18,
            moving_forward: false,
            moving_back: false,
            moving_left: false,
            moving_right: false,
            moving_down: false,
            moving_up: false,
            sprinting: false,
            mouse_delta: [0.0; 2],
        }
    }
}

impl FreeCamera {
    pub fn set_key(&mut self, key: SceneKey, pressed: bool) {
        match key {
            SceneKey::KeyW => self.moving_forward = pressed,
            SceneKey::KeyS => self.moving_back = pressed,
            SceneKey::KeyA => self.moving_left = pressed,
            SceneKey::KeyD => self.moving_right = pressed,
            SceneKey::ShiftLeft | SceneKey::KeyQ => self.moving_down = pressed,
            SceneKey::Space | SceneKey::KeyE => self.moving_up = pressed,
            SceneKey::ControlLeft => self.sprinting = pressed,
            _ => {}
        }
    }

    pub fn add_mouse_delta(&mut self, delta: [f32; 2]) {
        self.mouse_delta[0] += delta[0];
        self.mouse_delta[1] += delta[1];
    }

    pub fn adjust_speed_multiplier(&mut self, wheel_delta: f32) {
        if wheel_delta.abs() <= f32::EPSILON {
            return;
        }

        self.speed_multiplier =
            (self.speed_multiplier * 1.18_f32.powf(wheel_delta)).clamp(0.01, 120.0);
    }

    pub fn stop(&mut self) {
        self.velocity = [0.0; 3];
    }

    pub fn frame_sphere(&mut self, center: [f32; 3], radius: f32) {
        let radius = radius.max(1.0);
        let eye = [
            center[0],
            center[1] + radius * 0.24,
            center[2] + radius * 2.45,
        ];
        let forward =
            normalize_or_zero([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]);

        self.position = eye;
        self.velocity = [0.0; 3];
        self.yaw = forward[0].atan2(forward[2]);
        self.pitch = forward[1].asin().clamp(-1.35, 1.35);
        self.speed_multiplier = (radius / 28.0).clamp(0.15, 40.0);
    }

    pub fn update(&mut self, delta_time: f32) {
        let mouse_sensitivity = 0.0022;
        let move_speed = if self.sprinting { 8.5 } else { 5.2 } * self.speed_multiplier;
        let acceleration = 28.0;
        let damping = 14.0;

        self.yaw -= self.mouse_delta[0] * mouse_sensitivity;
        self.pitch -= self.mouse_delta[1] * mouse_sensitivity;
        self.mouse_delta = [0.0; 2];
        self.pitch = self.pitch.clamp(-1.35, 1.35);

        let forward = self.flat_forward();
        let right = normalize_or_zero(cross(forward, [0.0, 1.0, 0.0]));
        let mut input = [0.0; 3];

        if self.moving_forward {
            add_scaled(&mut input, forward, 1.0);
        }
        if self.moving_back {
            add_scaled(&mut input, forward, -1.0);
        }
        if self.moving_right {
            add_scaled(&mut input, right, 1.0);
        }
        if self.moving_left {
            add_scaled(&mut input, right, -1.0);
        }
        if self.moving_up {
            input[1] += 1.0;
        }
        if self.moving_down {
            input[1] -= 1.0;
        }

        let desired_velocity = scaled(normalize_or_zero(input), move_speed);
        let response = if input == [0.0; 3] {
            damping
        } else {
            acceleration
        };
        let blend = 1.0 - (-response * delta_time).exp();
        self.velocity = lerp3(self.velocity, desired_velocity, blend);
        add_scaled(&mut self.position, self.velocity, delta_time);
    }

    pub fn camera(&self) -> Camera {
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

    fn flat_forward(&self) -> [f32; 3] {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();

        [yaw_sin, 0.0, yaw_cos]
    }
}

fn add_scaled(target: &mut [f32; 3], value: [f32; 3], scale: f32) {
    target[0] += value[0] * scale;
    target[1] += value[1] * scale;
    target[2] += value[2] * scale;
}

fn scaled(value: [f32; 3], scale: f32) -> [f32; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
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
