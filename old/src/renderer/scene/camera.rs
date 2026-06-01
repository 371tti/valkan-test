use super::transform::{Mat4, look_at, mat4_mul, perspective, perspective_for_cubemap};

#[derive(Debug, Clone, Copy)]
pub struct CameraResponse {
    pub enabled: bool,
    pub exposure: f32,
    pub white_balance: [f32; 3],
    pub contrast: f32,
    pub saturation: f32,
}

impl CameraResponse {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            exposure: 1.0,
            white_balance: [1.0; 3],
            contrast: 1.0,
            saturation: 1.0,
        }
    }

    pub const fn enabled(exposure: f32, white_balance: [f32; 3]) -> Self {
        Self {
            enabled: true,
            exposure,
            white_balance,
            contrast: 1.06,
            saturation: 1.04,
        }
    }
}

impl Default for CameraResponse {
    fn default() -> Self {
        Self::disabled()
    }
}

pub const DEFAULT_CAMERA_FAR: f32 = 5_000.0;

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        mat4_mul(
            perspective(self.fov_y, aspect, self.near, self.far),
            look_at(self.eye, self.target, self.up),
        )
    }

    pub fn cubemap_view_projection(&self) -> Mat4 {
        mat4_mul(
            perspective_for_cubemap(self.fov_y, 1.0, self.near, self.far),
            look_at(self.eye, self.target, self.up),
        )
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: [0.0, 0.0, 4.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y: 60.0_f32.to_radians(),
            near: 0.1,
            far: DEFAULT_CAMERA_FAR,
        }
    }
}
