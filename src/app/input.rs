use winit::keyboard::{KeyCode, PhysicalKey};

use crate::renderer::SceneKey;

pub(super) fn scene_key(physical_key: PhysicalKey) -> SceneKey {
    match physical_key {
        PhysicalKey::Code(KeyCode::Escape) => SceneKey::Escape,
        PhysicalKey::Code(KeyCode::Space) => SceneKey::Space,
        PhysicalKey::Code(KeyCode::ShiftLeft) => SceneKey::ShiftLeft,
        PhysicalKey::Code(KeyCode::ControlLeft) => SceneKey::ControlLeft,
        PhysicalKey::Code(KeyCode::ArrowUp) => SceneKey::ArrowUp,
        PhysicalKey::Code(KeyCode::ArrowDown) => SceneKey::ArrowDown,
        PhysicalKey::Code(KeyCode::ArrowLeft) => SceneKey::ArrowLeft,
        PhysicalKey::Code(KeyCode::ArrowRight) => SceneKey::ArrowRight,
        PhysicalKey::Code(KeyCode::KeyW) => SceneKey::KeyW,
        PhysicalKey::Code(KeyCode::KeyA) => SceneKey::KeyA,
        PhysicalKey::Code(KeyCode::KeyS) => SceneKey::KeyS,
        PhysicalKey::Code(KeyCode::KeyD) => SceneKey::KeyD,
        PhysicalKey::Code(KeyCode::KeyQ) => SceneKey::KeyQ,
        PhysicalKey::Code(KeyCode::KeyE) => SceneKey::KeyE,
        PhysicalKey::Code(KeyCode::F12) => SceneKey::F12,
        _ => SceneKey::Other,
    }
}

pub(super) fn is_escape(physical_key: PhysicalKey) -> bool {
    physical_key == PhysicalKey::Code(KeyCode::Escape)
}
