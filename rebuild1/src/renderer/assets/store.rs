use std::collections::BTreeMap;

use crate::{
    import::ImportedScene,
    protocol::{
        AssetHandle, LoadedAsset, MaterialDescriptor, MaterialHandle, MeshHandle, SceneHandle,
        TextureDescriptor, TextureHandle,
    },
};

use super::{
    garbage::DeferredDestroyQueue, material::GpuMaterialAsset, mesh::GpuMeshAsset,
    texture::GpuTextureAsset,
};

pub(crate) struct GpuAssetStore {
    next_scene_raw: u64,
    next_mesh_raw: u64,
    next_material_raw: u64,
    next_texture_raw: u64,
    scenes: BTreeMap<SceneHandle, SceneAsset>,
    meshes: BTreeMap<MeshHandle, GpuMeshAsset>,
    materials: BTreeMap<MaterialHandle, GpuMaterialAsset>,
    textures: BTreeMap<TextureHandle, GpuTextureAsset>,
    garbage: DeferredDestroyQueue,
}

impl Default for GpuAssetStore {
    /// Starts the protocol handle counters at one so zero never leaves the store.
    fn default() -> Self {
        Self {
            next_scene_raw: 1,
            next_mesh_raw: 1,
            next_material_raw: 1,
            next_texture_raw: 1,
            scenes: BTreeMap::new(),
            meshes: BTreeMap::new(),
            materials: BTreeMap::new(),
            textures: BTreeMap::new(),
            garbage: DeferredDestroyQueue::default(),
        }
    }
}

impl GpuAssetStore {
    /// Allocates protocol handles for one imported scene without creating fallback assets.
    pub(crate) fn upload_imported_scene(&mut self, imported: &ImportedScene) -> LoadedAsset {
        let scene = self.alloc_scene();
        let meshes = self.upload_meshes(imported.meshes());
        let textures = self.upload_textures(imported.textures());
        let materials = self.upload_materials(imported.materials(), &textures);

        self.scenes.insert(
            scene,
            SceneAsset {
                meshes: meshes.clone(),
                materials: materials.clone(),
                textures: textures.clone(),
            },
        );

        tracing::trace!(
            source = %imported.source().display(),
            scene = scene.raw(),
            meshes = meshes.len(),
            materials = materials.len(),
            textures = textures.len(),
            "registered imported asset handles"
        );

        LoadedAsset::new(Some(scene), meshes, materials, textures, imported.bounds())
    }

    /// Invalidates one asset handle and defers destruction until GPU lifetime retirement.
    pub(crate) fn unload(&mut self, asset: AssetHandle) -> bool {
        match asset {
            AssetHandle::Scene(scene) => self.unload_scene(scene),
            AssetHandle::Mesh(mesh) => self.retire_mesh(mesh),
            AssetHandle::Material(material) => self.retire_material(material),
            AssetHandle::Texture(texture) => self.retire_texture(texture),
        }
    }

    /// Collects retired handles that are ready for backend-local GPU destruction.
    pub(crate) fn collect_deferred_destroys(&mut self) -> Vec<AssetHandle> {
        let retired = self.garbage.collect_ready();
        for destroy in &retired {
            tracing::trace!(
                asset = ?destroy.asset(),
                "collected deferred asset destroy"
            );
        }
        retired.into_iter().map(|destroy| destroy.asset()).collect()
    }

    /// Returns how many retired assets are waiting for GPU-safe destruction.
    pub(crate) fn pending_destroy_count(&self) -> usize {
        self.garbage.len()
    }

    /// Returns whether all mesh and material handles referenced by a draw are active.
    pub(crate) fn can_draw(&self, mesh: MeshHandle, material: MaterialHandle) -> bool {
        let Some(material) = self.materials.get(&material) else {
            return false;
        };
        self.meshes
            .get(&mesh)
            .is_some_and(GpuMeshAsset::is_draw_ready)
            && material.is_draw_ready(&self.textures)
    }

    /// Copies texture payload descriptors for handles returned by a single asset load.
    pub(crate) fn texture_descriptors(
        &self,
        handles: &[TextureHandle],
    ) -> Vec<(TextureHandle, TextureDescriptor)> {
        handles
            .iter()
            .filter_map(|handle| {
                self.textures
                    .get(handle)
                    .map(|texture| (*handle, texture.descriptor().clone()))
            })
            .collect()
    }

    /// Copies material descriptors for handles returned by a single asset load.
    pub(crate) fn material_descriptors(
        &self,
        handles: &[MaterialHandle],
    ) -> Vec<(MaterialHandle, MaterialDescriptor)> {
        handles
            .iter()
            .filter_map(|handle| {
                self.materials
                    .get(handle)
                    .map(|material| (*handle, material.descriptor().clone()))
            })
            .collect()
    }

    /// Returns whether a mesh handle is still active in the store.
    #[cfg(test)]
    pub(crate) fn contains_mesh(&self, mesh: MeshHandle) -> bool {
        self.meshes.contains_key(&mesh)
    }

    /// Returns whether a material handle is still active in the store.
    #[cfg(test)]
    pub(crate) fn contains_material(&self, material: MaterialHandle) -> bool {
        self.materials.contains_key(&material)
    }

    /// Returns whether a texture handle is still active in the store.
    #[cfg(test)]
    pub(crate) fn contains_texture(&self, texture: TextureHandle) -> bool {
        self.textures.contains_key(&texture)
    }

    /// Returns whether a scene handle is still active in the store.
    #[cfg(test)]
    pub(crate) fn contains_scene(&self, scene: SceneHandle) -> bool {
        self.scenes.contains_key(&scene)
    }

    /// Allocates one scene handle for imported scene ownership.
    fn alloc_scene(&mut self) -> SceneHandle {
        let handle = SceneHandle::from_raw(self.next_scene_raw)
            .expect("asset handle counter never yields zero");
        self.next_scene_raw = next_non_zero_raw(self.next_scene_raw);
        handle
    }

    /// Uploads imported mesh geometry and returns handles in imported mesh order.
    fn upload_meshes(&mut self, meshes: &[crate::import::ImportedMesh]) -> Vec<MeshHandle> {
        meshes
            .iter()
            .map(|mesh| {
                let handle = MeshHandle::from_raw(self.next_mesh_raw)
                    .expect("asset handle counter never yields zero");
                self.next_mesh_raw = next_non_zero_raw(self.next_mesh_raw);
                self.meshes
                    .insert(handle, GpuMeshAsset::from_imported(mesh));
                handle
            })
            .collect()
    }

    /// Uploads imported texture descriptors and returns handles in imported texture order.
    fn upload_textures(
        &mut self,
        textures: &[crate::import::ImportedTexture],
    ) -> Vec<TextureHandle> {
        textures
            .iter()
            .map(|texture| {
                let handle = self.alloc_texture_handle();
                self.textures
                    .insert(handle, GpuTextureAsset::from_imported(texture));
                handle
            })
            .collect()
    }

    /// Uploads imported material descriptors after resolving texture indices to handles.
    fn upload_materials(
        &mut self,
        materials: &[crate::import::ImportedMaterial],
        textures: &[TextureHandle],
    ) -> Vec<MaterialHandle> {
        materials
            .iter()
            .map(|material| {
                let handle = self.alloc_material_handle();
                self.materials
                    .insert(handle, GpuMaterialAsset::from_imported(material, textures));
                handle
            })
            .collect()
    }

    /// Allocates one material handle and advances the non-zero counter.
    fn alloc_material_handle(&mut self) -> MaterialHandle {
        let handle = MaterialHandle::from_raw(self.next_material_raw)
            .expect("asset handle counter never yields zero");
        self.next_material_raw = next_non_zero_raw(self.next_material_raw);
        handle
    }

    /// Allocates one texture handle and advances the non-zero counter.
    fn alloc_texture_handle(&mut self) -> TextureHandle {
        let handle = TextureHandle::from_raw(self.next_texture_raw)
            .expect("asset handle counter never yields zero");
        self.next_texture_raw = next_non_zero_raw(self.next_texture_raw);
        handle
    }

    /// Retires a scene and every child handle tracked through that scene load.
    fn unload_scene(&mut self, scene: SceneHandle) -> bool {
        let Some(asset) = self.scenes.remove(&scene) else {
            return false;
        };

        self.garbage.defer(AssetHandle::Scene(scene));
        for mesh in asset.meshes {
            self.retire_mesh(mesh);
        }
        for material in asset.materials {
            self.retire_material(material);
        }
        for texture in asset.textures {
            self.retire_texture(texture);
        }
        true
    }

    /// Removes one active mesh handle and queues its GPU resources for later destruction.
    fn retire_mesh(&mut self, mesh: MeshHandle) -> bool {
        let removed = self.meshes.remove(&mesh);
        if removed.is_some() {
            self.garbage.defer(AssetHandle::Mesh(mesh));
        }
        removed.is_some()
    }

    /// Removes one active material handle and queues its GPU resources for later destruction.
    fn retire_material(&mut self, material: MaterialHandle) -> bool {
        let removed = self.materials.remove(&material);
        if removed.is_some() {
            self.garbage.defer(AssetHandle::Material(material));
        }
        removed.is_some()
    }

    /// Removes one active texture handle and queues its GPU resources for later destruction.
    fn retire_texture(&mut self, texture: TextureHandle) -> bool {
        let removed = self.textures.remove(&texture);
        if removed.is_some() {
            self.garbage.defer(AssetHandle::Texture(texture));
        }
        removed.is_some()
    }
}

#[derive(Clone, Debug)]
struct SceneAsset {
    meshes: Vec<MeshHandle>,
    materials: Vec<MaterialHandle>,
    textures: Vec<TextureHandle>,
}

/// Returns the next non-zero counter value for protocol asset handles.
fn next_non_zero_raw(raw: u64) -> u64 {
    raw.checked_add(1).unwrap_or(1).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that unloading a scene invalidates all handles without immediate destruction.
    #[test]
    fn unload_scene_defers_child_asset_destroys() {
        let imported = ImportedScene::new("scene.r1scene".into(), 1, 1, 1);
        let mut store = GpuAssetStore::default();
        let loaded = store.upload_imported_scene(&imported);
        let scene = loaded.scene.expect("scene handle is allocated");
        let mesh = loaded.meshes[0];
        let material = loaded.materials[0];
        let texture = loaded.textures[0];

        assert!(store.unload(AssetHandle::Scene(scene)));

        assert!(!store.contains_scene(scene));
        assert!(!store.contains_mesh(mesh));
        assert!(!store.contains_material(material));
        assert!(!store.contains_texture(texture));
        assert_eq!(store.pending_destroy_count(), 4);
    }

    // Verifies that Stage 6 material descriptors preserve alpha mode and named texture slots.
    #[test]
    fn upload_textured_cutout_scene_keeps_material_slots() {
        let texture = crate::import::ImportedTexture::solid([255, 255, 255, 255]);
        let material = crate::import::ImportedMaterial::new(
            crate::protocol::MaterialAlphaMode::Cutout,
            350,
            vec![crate::import::ImportedMaterialTextureSlot::new(
                crate::protocol::MaterialTextureSlot::BaseColor,
                0,
            )],
        );
        let imported = ImportedScene::from_parts(
            "scene.r1scene".into(),
            vec![crate::import::ImportedMesh::Plane],
            vec![material],
            vec![texture],
        );
        let mut store = GpuAssetStore::default();

        let loaded = store.upload_imported_scene(&imported);
        let material = loaded.materials[0];
        let texture = loaded.textures[0];
        let stored = store
            .materials
            .get(&material)
            .expect("material should be stored");

        assert_eq!(
            stored.descriptor().alpha_mode(),
            crate::protocol::MaterialAlphaMode::Cutout
        );
        assert_eq!(
            stored
                .descriptor()
                .texture(crate::protocol::MaterialTextureSlot::BaseColor),
            Some(texture)
        );
    }

    // Verifies that Stage 7 starts storing real mesh geometry for future Vulkan upload.
    #[test]
    fn upload_plane_scene_keeps_mesh_geometry() {
        let imported = ImportedScene::from_parts(
            "scene.r1scene".into(),
            vec![crate::import::ImportedMesh::Plane],
            Vec::new(),
            Vec::new(),
        );
        let mut store = GpuAssetStore::default();

        let loaded = store.upload_imported_scene(&imported);
        let mesh = loaded.meshes[0];
        let stored = store.meshes.get(&mesh).expect("mesh should be stored");

        assert_eq!(stored.geometry().vertex_count(), 4);
        assert_eq!(stored.geometry().index_count(), 6);
        assert!(stored.is_draw_ready());
    }
}
