use crate::import::ImportedMesh;

#[derive(Clone, Debug)]
pub(crate) struct GpuMeshAsset {
    geometry: MeshGeometry,
}

impl GpuMeshAsset {
    /// Builds renderer mesh geometry from the importer-owned intermediate mesh record.
    pub(crate) fn from_imported(imported: &ImportedMesh) -> Self {
        let geometry = MeshGeometry::from_imported(imported);
        tracing::trace!(
            vertices = geometry.vertex_count(),
            indices = geometry.index_count(),
            "registered mesh geometry"
        );

        Self { geometry }
    }

    /// Returns whether this mesh has enough geometry for indexed triangle rendering.
    pub(crate) fn is_draw_ready(&self) -> bool {
        self.geometry.vertex_count() >= 3 && self.geometry.index_count() >= 3
    }

    /// Returns the immutable geometry payload kept for the Vulkan mesh uploader.
    #[cfg(test)]
    pub(crate) fn geometry(&self) -> &MeshGeometry {
        &self.geometry
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MeshGeometry {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
}

impl MeshGeometry {
    /// Converts one imported mesh shape into renderer-owned vertices and indices.
    pub(crate) fn from_imported(imported: &ImportedMesh) -> Self {
        match imported {
            ImportedMesh::Plane => Self::plane(),
            ImportedMesh::Indexed(data) => Self {
                vertices: data
                    .vertices()
                    .iter()
                    .map(|vertex| {
                        MeshVertex::new(
                            vertex.position(),
                            vertex.normal(),
                            vertex.uv(),
                            vertex.color(),
                        )
                    })
                    .collect(),
                indices: data.indices().to_vec(),
            },
        }
    }

    /// Creates the explicit plane geometry requested by `.r1scene` manifests.
    fn plane() -> Self {
        Self {
            vertices: vec![
                MeshVertex::new(
                    [-0.75, -0.55, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.25, 0.20, 1.0],
                ),
                MeshVertex::new(
                    [0.75, -0.55, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0],
                    [0.20, 0.85, 0.45, 1.0],
                ),
                MeshVertex::new(
                    [0.75, 0.55, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0],
                    [0.20, 0.45, 1.0, 1.0],
                ),
                MeshVertex::new(
                    [-0.75, 0.55, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 0.85, 0.20, 1.0],
                ),
            ],
            indices: vec![0, 1, 2, 2, 3, 0],
        }
    }

    /// Returns the number of vertices available for upload.
    pub(crate) fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Returns vertices in the memory layout expected by the Vulkan mesh uploader.
    pub(crate) fn vertices(&self) -> &[MeshVertex] {
        &self.vertices
    }

    /// Returns the number of indices available for indexed drawing.
    pub(crate) fn index_count(&self) -> usize {
        self.indices.len()
    }

    /// Returns indices in the order used by indexed triangle rendering.
    pub(crate) fn indices(&self) -> &[u32] {
        &self.indices
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MeshVertex {
    pub(crate) position: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) uv: [f32; 2],
    pub(crate) color: [f32; 4],
}

impl MeshVertex {
    /// Creates one renderer-owned vertex in the exact memory layout used by mesh shaders.
    pub(crate) fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], color: [f32; 4]) -> Self {
        Self {
            position,
            normal,
            uv,
            color,
        }
    }
}
