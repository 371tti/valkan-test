#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub ambient: [f32; 3],
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [-0.35, -0.75, -0.55],
            color: [1.0, 0.94, 0.86],
            intensity: 1.2,
            ambient: [0.035, 0.04, 0.055],
        }
    }
}
