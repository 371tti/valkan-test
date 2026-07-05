use gr_render::protocol::{CameraSnapshot, SceneBounds};

const CAMERA_SMOOTHING_SECONDS: [f32; 7] = [0.0, 0.025, 0.045, 0.070, 0.105, 0.150, 0.220];
const DEFAULT_CAMERA_SMOOTHING_LEVEL: usize = 2;
const DEFAULT_NEAR_PLANE: f32 = 0.06;
const DEFAULT_FAR_PLANE: f32 = 1500.0;
const MIN_FAR_PLANE: f32 = 32.0;
const MAX_FAR_PLANE: f32 = 5000.0;
const SCENE_FAR_RADIUS_MARGIN: f32 = 1.35;
const SCENE_FAR_ABSOLUTE_MARGIN: f32 = 32.0;

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
    stabilized_position: [f32; 3],
    stabilized_yaw: f32,
    stabilized_pitch: f32,
    scene_bounds: Option<SceneBounds>,
    smoothing_level: usize,
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
        let position = [0.0, 1.8, 5.0];
        let yaw = std::f32::consts::PI;
        let pitch = -0.18;

        Self {
            position,
            velocity: [0.0; 3],
            speed_multiplier: 1.0,
            yaw,
            pitch,
            stabilized_position: position,
            stabilized_yaw: yaw,
            stabilized_pitch: pitch,
            scene_bounds: None,
            smoothing_level: DEFAULT_CAMERA_SMOOTHING_LEVEL,
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

    /// Adjusts camera stabilization strength and returns whether the level changed.
    pub(super) fn adjust_smoothing_level(&mut self, delta: i32) -> bool {
        let old = self.smoothing_level;
        let max = CAMERA_SMOOTHING_SECONDS.len() as i32 - 1;
        self.smoothing_level = (old as i32 + delta).clamp(0, max) as usize;
        if self.smoothing_level == 0 {
            self.reset_stabilized_pose();
        }

        self.smoothing_level != old
    }

    /// Returns the current camera stabilization time constant.
    pub(super) fn smoothing_seconds(&self) -> f32 {
        CAMERA_SMOOTHING_SECONDS[self.smoothing_level]
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
        self.scene_bounds = Some(bounds);
        self.reset_stabilized_pose();
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
        self.update_stabilized_pose(delta_time);
    }

    /// Extracts the camera as owned protocol data for the renderer thread.
    pub(super) fn snapshot(&self) -> CameraSnapshot {
        camera_snapshot(
            self.stabilized_position,
            self.stabilized_yaw,
            self.stabilized_pitch,
            self.scene_bounds,
        )
    }

    /// Stops inertial motion without changing the current camera orientation.
    fn stop(&mut self) {
        self.velocity = [0.0; 3];
    }

    /// Snaps the renderer-facing camera to the input camera without lingering interpolation.
    fn reset_stabilized_pose(&mut self) {
        self.stabilized_position = self.position;
        self.stabilized_yaw = self.yaw;
        self.stabilized_pitch = self.pitch;
    }

    /// Smooths tiny input jitter before the camera snapshot crosses into the renderer.
    fn update_stabilized_pose(&mut self, delta_time: f32) {
        let smoothing_seconds = self.smoothing_seconds();
        if smoothing_seconds <= f32::EPSILON {
            self.reset_stabilized_pose();
            return;
        }

        let delta_time = delta_time.max(0.0);
        if delta_time <= f32::EPSILON {
            return;
        }

        let blend = 1.0 - (-(delta_time / smoothing_seconds)).exp();
        self.stabilized_position = lerp3(self.stabilized_position, self.position, blend);
        self.stabilized_yaw = lerp_angle(self.stabilized_yaw, self.yaw, blend);
        self.stabilized_pitch = lerp(self.stabilized_pitch, self.pitch, blend);
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

    /// Returns the horizontal forward vector used for WASD movement.
    fn flat_forward(&self) -> [f32; 3] {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();

        [yaw_sin, 0.0, yaw_cos]
    }
}

/// Builds a validated protocol camera from compact app-side pose values.
fn camera_snapshot(
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    scene_bounds: Option<SceneBounds>,
) -> CameraSnapshot {
    let forward = forward_from_angles(yaw, pitch);
    let (near, far) = clip_planes_for_scene(position, forward, scene_bounds);

    CameraSnapshot::perspective(
        position,
        [
            position[0] + forward[0],
            position[1] + forward[1],
            position[2] + forward[2],
        ],
        [0.0, 1.0, 0.0],
        60.0_f32.to_radians(),
        near,
        far,
    )
    .expect("free camera keeps finite values and ordered clip planes")
}

/// Returns tighter clip planes so depth precision is spent on the visible model volume.
fn clip_planes_for_scene(
    position: [f32; 3],
    forward: [f32; 3],
    scene_bounds: Option<SceneBounds>,
) -> (f32, f32) {
    let Some(bounds) = scene_bounds else {
        return (DEFAULT_NEAR_PLANE, DEFAULT_FAR_PLANE);
    };

    let radius = bounds.radius().max(1.0);
    let to_center = sub(bounds.center(), position);
    let center_depth = dot(to_center, forward);
    let scene_front_depth = center_depth + radius * SCENE_FAR_RADIUS_MARGIN;
    let far = (scene_front_depth + SCENE_FAR_ABSOLUTE_MARGIN).clamp(MIN_FAR_PLANE, MAX_FAR_PLANE);
    let near = (radius * 0.00035).clamp(DEFAULT_NEAR_PLANE, 0.16);

    (near, far.max(near + 1.0))
}

/// Converts yaw/pitch into the full camera forward vector.
fn forward_from_angles(yaw: f32, pitch: f32) -> [f32; 3] {
    let (yaw_sin, yaw_cos) = yaw.sin_cos();
    let (pitch_sin, pitch_cos) = pitch.sin_cos();

    [yaw_sin * pitch_cos, pitch_sin, yaw_cos * pitch_cos]
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

/// Interpolates one scalar value.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Interpolates angles using the shortest yaw arc to avoid wraparound jumps.
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    a + shortest_angle_delta(a, b) * t
}

/// Returns the signed shortest angle delta from `from` to `to`.
fn shortest_angle_delta(from: f32, to: f32) -> f32 {
    let two_pi = std::f32::consts::PI * 2.0;
    let delta = (to - from + std::f32::consts::PI).rem_euclid(two_pi) - std::f32::consts::PI;

    if delta <= -std::f32::consts::PI {
        delta + two_pi
    } else {
        delta
    }
}

/// Normalizes a vector and returns zero when no direction is available.
fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

    if len <= f32::EPSILON {
        return [0.0; 3];
    }

    [v[0] / len, v[1] / len, v[2] / len]
}

/// Returns the difference between two points.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Returns the dot product for compact camera-space tests.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Returns the cross product used to derive camera strafe direction.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_smoothing_lags_large_single_frame_mouse_jitter() {
        let mut camera = FreeCamera::default();
        camera.adjust_smoothing_level(CAMERA_SMOOTHING_SECONDS.len() as i32);
        let initial_yaw = camera.stabilized_yaw;

        camera.add_mouse_delta([-600.0, 0.0]);
        camera.update(1.0 / 120.0);

        assert!(camera.yaw > initial_yaw + 1.0);
        assert!(camera.stabilized_yaw > initial_yaw);
        assert!(camera.stabilized_yaw < camera.yaw);
    }

    #[test]
    fn disabling_camera_smoothing_snaps_to_input_pose() {
        let mut camera = FreeCamera::default();
        camera.add_mouse_delta([-600.0, 0.0]);
        camera.update(1.0 / 120.0);

        assert!(camera.stabilized_yaw < camera.yaw);
        camera.adjust_smoothing_level(-(CAMERA_SMOOTHING_SECONDS.len() as i32));

        assert_eq!(camera.smoothing_seconds(), 0.0);
        assert_eq!(camera.stabilized_position, camera.position);
        assert_eq!(camera.stabilized_yaw, camera.yaw);
        assert_eq!(camera.stabilized_pitch, camera.pitch);
    }

    #[test]
    fn framing_bounds_resets_camera_smoothing() {
        let mut camera = FreeCamera::default();
        camera.add_mouse_delta([-600.0, 0.0]);
        camera.update(1.0 / 120.0);
        let bounds = SceneBounds::new([2.0, 3.0, -4.0], 8.0).expect("finite bounds");

        camera.frame_bounds(bounds);

        assert_eq!(camera.stabilized_position, camera.position);
        assert_eq!(camera.stabilized_yaw, camera.yaw);
        assert_eq!(camera.stabilized_pitch, camera.pitch);
    }

    #[test]
    fn default_camera_uses_bounded_clip_planes() {
        let snapshot = FreeCamera::default().snapshot();

        assert_eq!(snapshot.near, DEFAULT_NEAR_PLANE);
        assert_eq!(snapshot.far, DEFAULT_FAR_PLANE);
    }

    #[test]
    fn framed_camera_tightens_clip_planes_around_scene_bounds() {
        let mut camera = FreeCamera::default();
        let bounds = SceneBounds::new([0.0, 0.0, 0.0], 300.0).expect("finite bounds");

        camera.frame_bounds(bounds);
        let snapshot = camera.snapshot();
        let forward = normalize_or_zero(sub(bounds.center(), snapshot.eye));
        let center_depth = dot(sub(bounds.center(), snapshot.eye), forward);

        assert!(snapshot.near > DEFAULT_NEAR_PLANE);
        assert!(snapshot.far < DEFAULT_FAR_PLANE);
        assert!(snapshot.far > center_depth + bounds.radius());
    }
}
