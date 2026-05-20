#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub usize);

impl MeshId {
    pub const CUBE: Self = Self(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub usize);

impl MaterialId {
    pub const DEFAULT: Self = Self(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub usize);

impl TextureId {
    pub const DEFAULT: Self = Self(0);
    pub const NORMAL: Self = Self(1);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelId(pub usize);

impl ModelId {
    pub const CUBE: Self = Self(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(pub usize);

impl PipelineId {
    pub const LIT_MESH: Self = Self(0);
    pub const LIT_MESH_TRANSPARENT: Self = Self(1);
    pub const LIT_MESH_WIREFRAME: Self = Self(2);
    pub const LIT_MESH_DOUBLE_SIDED: Self = Self(3);
    pub const LIT_MESH_TRANSPARENT_DOUBLE_SIDED: Self = Self(4);
}
