use valkan_test::renderer::{Camera, SceneKey};

use super::math::{add_scaled, cross, lerp3, normalize, normalize_or_zero, scale};

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
            (self.speed_multiplier * 1.18_f32.powf(wheel_delta)).clamp(0.15, 12.0);
    }

    pub fn stop(&mut self) {
        self.velocity = [0.0; 3];
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
        let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
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

        let desired_velocity = scale(normalize_or_zero(input), move_speed);
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
