use gr_render::protocol::{CameraSnapshot, SceneBounds};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CameraKey {
    Forward,
    Back,
    Left,
    Right,
    Down,
    Up,
    Sprint,
    Stop,
    Other,
}

#[derive(Debug, Clone)]
pub(super) struct FreeCamera {
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
    /// Creates the same early viewer pose used by the old demo camera.
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
    /// Updates one movement key flag without exposing winit input types to extraction code.
    pub(super) fn set_key(&mut self, key: CameraKey, pressed: bool) {
        match key {
            CameraKey::Forward => self.moving_forward = pressed,
            CameraKey::Back => self.moving_back = pressed,
            CameraKey::Left => self.moving_left = pressed,
            CameraKey::Right => self.moving_right = pressed,
            CameraKey::Down => self.moving_down = pressed,
            CameraKey::Up => self.moving_up = pressed,
            CameraKey::Sprint => self.sprinting = pressed,
            CameraKey::Stop if pressed => self.stop(),
            CameraKey::Stop | CameraKey::Other => {}
        }
    }

    /// Accumulates raw captured mouse movement until the next frame update consumes it.
    pub(super) fn add_mouse_delta(&mut self, delta: [f32; 2]) {
        self.mouse_delta[0] += delta[0];
        self.mouse_delta[1] += delta[1];
    }

    /// Changes movement speed by mouse wheel while keeping camera control bounded.
    pub(super) fn adjust_speed_multiplier(&mut self, wheel_delta: f32) {
        if wheel_delta.abs() <= f32::EPSILON {
            return;
        }

        self.speed_multiplier =
            (self.speed_multiplier * 1.18_f32.powf(wheel_delta)).clamp(0.01, 120.0);
    }

    /// Frames a newly loaded model so the first camera snapshot sees it immediately.
    pub(super) fn frame_bounds(&mut self, bounds: SceneBounds) {
        let center = bounds.center();
        let radius = bounds.radius().max(1.0);
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

    /// Advances camera rotation and movement for one app-side frame extraction.
    pub(super) fn update(&mut self, delta_time: f32) {
        let mouse_sensitivity = 0.0022;
        let move_speed = if self.sprinting { 8.5 } else { 5.2 } * self.speed_multiplier;
        let acceleration = 28.0;
        let damping = 14.0;

        self.yaw -= self.mouse_delta[0] * mouse_sensitivity;
        self.pitch -= self.mouse_delta[1] * mouse_sensitivity;
        self.mouse_delta = [0.0; 2];
        self.pitch = self.pitch.clamp(-1.35, 1.35);

        let input = self.input_vector();
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

    /// Extracts the camera as owned protocol data for the renderer thread.
    pub(super) fn snapshot(&self) -> CameraSnapshot {
        let forward = self.forward();

        CameraSnapshot::perspective(
            self.position,
            [
                self.position[0] + forward[0],
                self.position[1] + forward[1],
                self.position[2] + forward[2],
            ],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.03,
            5000.0,
        )
        .expect("free camera keeps finite values and ordered clip planes")
    }

    /// Stops inertial motion without changing the current camera orientation.
    fn stop(&mut self) {
        self.velocity = [0.0; 3];
    }

    /// Builds the desired movement vector from the current key state.
    fn input_vector(&self) -> [f32; 3] {
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

        input
    }

    /// Returns the full camera forward vector including pitch.
    fn forward(&self) -> [f32; 3] {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();

        [yaw_sin * pitch_cos, pitch_sin, yaw_cos * pitch_cos]
    }

    /// Returns the horizontal forward vector used for WASD movement.
    fn flat_forward(&self) -> [f32; 3] {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();

        [yaw_sin, 0.0, yaw_cos]
    }
}

/// Adds `value * scale` into `target` for compact movement integration.
fn add_scaled(target: &mut [f32; 3], value: [f32; 3], scale: f32) {
    target[0] += value[0] * scale;
    target[1] += value[1] * scale;
    target[2] += value[2] * scale;
}

/// Multiplies a vector by a scalar.
fn scaled(value: [f32; 3], scale: f32) -> [f32; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

/// Interpolates camera velocity for smooth start and stop behavior.
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Normalizes a vector and returns zero when no direction is available.
fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

    if len <= f32::EPSILON {
        return [0.0; 3];
    }

    [v[0] / len, v[1] / len, v[2] / len]
}

/// Returns the cross product used to derive camera strafe direction.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
