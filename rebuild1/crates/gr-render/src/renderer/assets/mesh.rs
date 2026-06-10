use crate::{import::ImportedMesh, math::length3, protocol::SceneBounds};

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

    /// Returns a finite bounding sphere used by renderer-side visibility decisions.
    pub(crate) fn bounds(&self) -> Option<SceneBounds> {
        let (min, max) = self.position_min_max()?;
        let center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let radius = self
            .vertices
            .iter()
            .map(|vertex| {
                length3([
                    vertex.position[0] - center[0],
                    vertex.position[1] - center[1],
                    vertex.position[2] - center[2],
                ])
            })
            .fold(0.0_f32, f32::max);

        SceneBounds::new(center, radius.max(0.001))
    }

    /// Returns min/max positions for finite mesh vertices.
    fn position_min_max(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        let mut found = false;

        for vertex in &self.vertices {
            if !vertex.position.iter().all(|value| value.is_finite()) {
                continue;
            }
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex.position[axis]);
                max[axis] = max[axis].max(vertex.position[axis]);
            }
            found = true;
        }

        found.then_some((min, max))
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

impl meshopt::DecodePosition for MeshVertex {
    /// Exposes mesh positions to meshoptimizer without coupling LOD generation to Vulkan.
    fn decode_position(&self) -> [f32; 3] {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that mesh geometry exposes bounds for renderer-side culling and LOD.
    #[test]
    fn mesh_geometry_reports_bounds() {
        let geometry = MeshGeometry::plane();
        let bounds = geometry.bounds().expect("plane bounds should exist");

        assert_eq!(bounds.center(), [0.0, 0.0, 0.0]);
        assert!(bounds.radius() > 0.9);
    }
}
