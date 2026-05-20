use super::{MaterialId, MeshId, ModelId, PipelineId, Transform};

#[derive(Debug, Clone, Copy)]
pub struct RenderObject {
    pub mesh: MeshId,
    pub pipeline: PipelineId,
    pub transform: Transform,
    pub material: MaterialId,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderModel {
    pub model: ModelId,
    pub pipeline: PipelineId,
    pub transform: Transform,
}
