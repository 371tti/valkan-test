use std::{collections::BTreeMap, mem::size_of};

use ash::vk::Handle;
use ash::{Device, vk};

use crate::{
    protocol::{
        AssetHandle, MaterialAlphaMode, MaterialDescriptor, MaterialHandle, MaterialTextureSlot,
        TextureDescriptor, TextureHandle,
    },
    renderer::pipeline::shader_interface,
};

use super::{
    VulkanError,
    buffer::{GpuBuffer, create_device_local_buffer_with_data},
    material_texture::{DefaultMaterialTextures, VulkanTexture},
};

const MATERIAL_DESCRIPTOR_CAPACITY: u32 = 1024;

pub(super) struct VulkanMaterialStore {
    textures: BTreeMap<TextureHandle, VulkanTexture>,
    materials: BTreeMap<MaterialHandle, VulkanMaterial>,
    defaults: Option<DefaultMaterialTextures>,
    descriptor_pool: vk::DescriptorPool,
    material_set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
}

#[derive(Clone, Copy)]
pub(super) struct MaterialDrawInfo {
    pub(super) descriptor_set: vk::DescriptorSet,
    pub(super) uses_any_texture: bool,
    pub(super) uses_base_color_texture: bool,
    pub(super) uses_shadow_alpha_texture: bool,
    pub(super) uses_shadow_alpha_test: bool,
    pub(super) double_sided: bool,
    pub(super) fully_opaque: bool,
    pub(super) transparent: bool,
    pub(super) casts_opaque_shadow: bool,
    pub(super) casts_translucent_shadow: bool,
}

impl VulkanMaterialStore {
    /// Creates Vulkan descriptor resources used by uploaded material records.
    pub(super) fn create(device: &Device) -> Result<Self, VulkanError> {
        let material_set_layout = create_material_set_layout(device)?;
        let descriptor_pool = match create_descriptor_pool(device) {
            Ok(pool) => pool,
            Err(error) => {
                destroy_descriptor_set_layout(device, material_set_layout);
                return Err(error);
            }
        };
        let sampler = match create_sampler(device) {
            Ok(sampler) => sampler,
            Err(error) => {
                destroy_descriptor_pool(device, descriptor_pool);
                destroy_descriptor_set_layout(device, material_set_layout);
                return Err(error);
            }
        };
        tracing::info!("created Vulkan material resources");

        Ok(Self {
            textures: BTreeMap::new(),
            materials: BTreeMap::new(),
            defaults: None,
            descriptor_pool,
            material_set_layout,
            sampler,
        })
    }

    /// Uploads imported RGBA texture payloads into sampled Vulkan images.
    pub(super) fn upload_imported_textures(
        &mut self,
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue_family_index: u32,
        queue: vk::Queue,
        records: &[(TextureHandle, TextureDescriptor)],
    ) -> Result<(), VulkanError> {
        if !records.is_empty() {
            self.ensure_default_textures(device, memory_properties, queue_family_index, queue)?;
        }
        for (handle, descriptor) in records {
            let texture = VulkanTexture::upload(
                device,
                memory_properties,
                queue_family_index,
                queue,
                descriptor,
            )?;
            tracing::trace!(
                texture = handle.raw(),
                width = descriptor.width(),
                height = descriptor.height(),
                format = descriptor.format().name(),
                "uploaded Vulkan texture image"
            );
            if let Some(old) = self.textures.insert(*handle, texture) {
                old.destroy(device);
            }
        }

        Ok(())
    }

    /// Uploads material parameter buffers and texture descriptor sets for imported materials.
    pub(super) fn upload_imported_materials(
        &mut self,
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue_family_index: u32,
        queue: vk::Queue,
        records: &[(MaterialHandle, MaterialDescriptor)],
    ) -> Result<(), VulkanError> {
        for (handle, descriptor) in records {
            let material = VulkanMaterial::upload(
                device,
                memory_properties,
                self.descriptor_pool,
                self.material_set_layout,
                self.sampler,
                queue_family_index,
                queue,
                descriptor,
                &self.textures,
                self.defaults.as_ref(),
            )?;
            tracing::trace!(
                material = handle.raw(),
                alpha_mode = descriptor.alpha_mode().name(),
                texture_count = descriptor.textures().len(),
                descriptor_set = material.descriptor_set.as_raw(),
                "uploaded Vulkan material descriptors"
            );
            if let Some(old) = self.materials.insert(*handle, material) {
                old.destroy(device);
            }
        }

        Ok(())
    }

    /// Returns the material descriptor set layout used by mesh pipeline layouts.
    pub(super) fn material_set_layout(&self) -> vk::DescriptorSetLayout {
        self.material_set_layout
    }

    /// Returns all hot-path material facts used for draw list building and command recording.
    pub(super) fn draw_info(&self, material: MaterialHandle) -> Option<MaterialDrawInfo> {
        self.materials.get(&material).map(VulkanMaterial::draw_info)
    }

    /// Destroys backend material and texture resources whose protocol handles have retired.
    pub(super) fn destroy_retired(&mut self, device: &Device, retired: &[AssetHandle]) {
        for asset in retired {
            match *asset {
                AssetHandle::Material(material) => self.destroy_material(device, material),
                AssetHandle::Texture(texture) => self.destroy_texture(device, texture),
                AssetHandle::Scene(_) | AssetHandle::Mesh(_) => {}
            }
        }
    }

    /// Destroys every uploaded material, texture image, and descriptor owner.
    pub(super) fn destroy(self, device: &Device) {
        for material in self.materials.into_values() {
            material.destroy(device);
        }
        for texture in self.textures.into_values() {
            texture.destroy(device);
        }
        if let Some(defaults) = self.defaults {
            defaults.destroy(device);
        }
        destroy_sampler(device, self.sampler);
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_descriptor_set_layout(device, self.material_set_layout);
    }

    /// Destroys one material parameter buffer when its handle retires.
    fn destroy_material(&mut self, device: &Device, material: MaterialHandle) {
        if let Some(uploaded) = self.materials.remove(&material) {
            tracing::trace!(
                material = material.raw(),
                "destroying retired Vulkan material"
            );
            uploaded.destroy(device);
        }
    }

    /// Destroys one sampled image when its texture handle retires.
    fn destroy_texture(&mut self, device: &Device, texture: TextureHandle) {
        if let Some(uploaded) = self.textures.remove(&texture) {
            tracing::trace!(texture = texture.raw(), "destroying retired Vulkan texture");
            uploaded.destroy(device);
        }
    }

    /// Creates the explicit glTF default maps used to fill optional material descriptor slots.
    fn ensure_default_textures(
        &mut self,
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue_family_index: u32,
        queue: vk::Queue,
    ) -> Result<(), VulkanError> {
        if self.defaults.is_some() {
            return Ok(());
        }

        self.defaults = Some(DefaultMaterialTextures::create(
            device,
            memory_properties,
            queue_family_index,
            queue,
        )?);
        tracing::trace!("created explicit Vulkan material default texture maps");

        Ok(())
    }
}

impl Default for VulkanMaterialStore {
    /// Creates an inert store for tests that do not construct Vulkan resources.
    fn default() -> Self {
        Self {
            textures: BTreeMap::new(),
            materials: BTreeMap::new(),
            defaults: None,
            descriptor_pool: vk::DescriptorPool::null(),
            material_set_layout: vk::DescriptorSetLayout::null(),
            sampler: vk::Sampler::null(),
        }
    }
}

struct VulkanMaterial {
    params: GpuBuffer,
    descriptor_set: vk::DescriptorSet,
    texture_flags: MaterialTextureFlags,
    alpha_mode: MaterialAlphaMode,
    double_sided: bool,
}

impl VulkanMaterial {
    /// Creates one material parameter buffer and writes descriptors for explicit textures only.
    fn upload(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        descriptor_pool: vk::DescriptorPool,
        material_set_layout: vk::DescriptorSetLayout,
        sampler: vk::Sampler,
        queue_family_index: u32,
        queue: vk::Queue,
        descriptor: &MaterialDescriptor,
        textures: &BTreeMap<TextureHandle, VulkanTexture>,
        defaults: Option<&DefaultMaterialTextures>,
    ) -> Result<Self, VulkanError> {
        let texture_flags = MaterialTextureFlags::from_descriptor(descriptor);
        let texture_set = MaterialTextureSet::resolve(descriptor, textures, defaults)?;
        let params = create_device_local_buffer_with_data(
            device,
            memory_properties,
            queue_family_index,
            queue,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &[MaterialParams::from_descriptor(descriptor, texture_flags)],
        )?;
        let descriptor_set =
            match allocate_material_descriptor_set(device, descriptor_pool, material_set_layout) {
                Ok(set) => set,
                Err(error) => {
                    params.destroy(device);
                    return Err(error);
                }
            };

        update_material_descriptor_set(device, descriptor_set, &params, sampler, texture_set);
        Ok(Self {
            params,
            descriptor_set,
            texture_flags,
            alpha_mode: descriptor.alpha_mode(),
            double_sided: descriptor.double_sided(),
        })
    }

    /// Packs material state needed by per-frame draw list preparation.
    fn draw_info(&self) -> MaterialDrawInfo {
        let transparent = matches!(self.alpha_mode, MaterialAlphaMode::Transparent);
        let uses_base_color_texture = self.texture_flags.has(MaterialTextureSlot::BaseColor);
        MaterialDrawInfo {
            descriptor_set: self.descriptor_set,
            uses_any_texture: self.texture_flags.any(),
            uses_base_color_texture,
            uses_shadow_alpha_texture: uses_base_color_texture
                && matches!(self.alpha_mode, MaterialAlphaMode::Cutout),
            uses_shadow_alpha_test: matches!(self.alpha_mode, MaterialAlphaMode::Cutout),
            double_sided: self.double_sided,
            fully_opaque: matches!(self.alpha_mode, MaterialAlphaMode::Opaque),
            transparent,
            casts_opaque_shadow: matches!(
                self.alpha_mode,
                MaterialAlphaMode::Opaque | MaterialAlphaMode::Cutout
            ),
            casts_translucent_shadow: transparent,
        }
    }

    /// Destroys the material parameter buffer after GPU use has retired.
    fn destroy(self, device: &Device) {
        self.params.destroy(device);
    }
}

#[derive(Clone, Copy)]
struct MaterialTextureFlags {
    bits: u32,
}

impl MaterialTextureFlags {
    const BASE_COLOR: u32 = 1 << 0;
    const NORMAL: u32 = 1 << 1;
    const METALLIC_ROUGHNESS: u32 = 1 << 2;
    const OCCLUSION: u32 = 1 << 3;
    const EMISSIVE: u32 = 1 << 4;

    /// Builds the shader texture bitmask from explicit material slots.
    fn from_descriptor(descriptor: &MaterialDescriptor) -> Self {
        let mut bits = 0;
        for slot in descriptor.textures().keys() {
            bits |= Self::bit(*slot);
        }

        Self { bits }
    }

    /// Returns whether a material provides at least one sampled texture map.
    fn any(self) -> bool {
        self.bits != 0
    }

    /// Returns whether a specific material slot is present.
    fn has(self, slot: MaterialTextureSlot) -> bool {
        self.bits & Self::bit(slot) != 0
    }

    /// Returns the shader-side bitmask value.
    fn bits(self) -> u32 {
        self.bits
    }

    /// Returns the stable bit assigned to one material texture slot.
    fn bit(slot: MaterialTextureSlot) -> u32 {
        match slot {
            MaterialTextureSlot::BaseColor => Self::BASE_COLOR,
            MaterialTextureSlot::Normal => Self::NORMAL,
            MaterialTextureSlot::MetallicRoughness => Self::METALLIC_ROUGHNESS,
            MaterialTextureSlot::Occlusion => Self::OCCLUSION,
            MaterialTextureSlot::Emissive => Self::EMISSIVE,
        }
    }
}

struct MaterialTextureSet<'a> {
    base_color: &'a VulkanTexture,
    normal: &'a VulkanTexture,
    metallic_roughness: &'a VulkanTexture,
    occlusion: &'a VulkanTexture,
    emissive: &'a VulkanTexture,
}

impl<'a> MaterialTextureSet<'a> {
    /// Resolves explicit handles and fills absent slots with glTF default maps for textured shaders.
    fn resolve(
        descriptor: &'a MaterialDescriptor,
        textures: &'a BTreeMap<TextureHandle, VulkanTexture>,
        defaults: Option<&'a DefaultMaterialTextures>,
    ) -> Result<Option<Self>, VulkanError> {
        if descriptor.textures().is_empty() {
            return Ok(None);
        }
        let defaults = defaults.ok_or_else(|| {
            VulkanError::ShaderInterface(
                "material texture defaults must exist before textured descriptor upload".into(),
            )
        })?;

        Ok(Some(Self {
            base_color: Self::slot_texture(
                descriptor,
                textures,
                defaults,
                MaterialTextureSlot::BaseColor,
            )?,
            normal: Self::slot_texture(
                descriptor,
                textures,
                defaults,
                MaterialTextureSlot::Normal,
            )?,
            metallic_roughness: Self::slot_texture(
                descriptor,
                textures,
                defaults,
                MaterialTextureSlot::MetallicRoughness,
            )?,
            occlusion: Self::slot_texture(
                descriptor,
                textures,
                defaults,
                MaterialTextureSlot::Occlusion,
            )?,
            emissive: Self::slot_texture(
                descriptor,
                textures,
                defaults,
                MaterialTextureSlot::Emissive,
            )?,
        }))
    }

    /// Returns an explicit texture handle or the slot's default map when the slot is absent.
    fn slot_texture(
        descriptor: &'a MaterialDescriptor,
        textures: &'a BTreeMap<TextureHandle, VulkanTexture>,
        defaults: &'a DefaultMaterialTextures,
        slot: MaterialTextureSlot,
    ) -> Result<&'a VulkanTexture, VulkanError> {
        let Some(handle) = descriptor.texture(slot) else {
            return Ok(defaults.texture(slot));
        };
        textures.get(&handle).ok_or_else(|| {
            VulkanError::ShaderInterface(format!(
                "material references texture handle {} before upload",
                handle.raw()
            ))
        })
    }

    /// Creates descriptor image infos for every material sampler binding.
    fn image_infos(&self, sampler: vk::Sampler) -> MaterialImageInfos {
        MaterialImageInfos {
            base_color: [texture_image_info(sampler, self.base_color)],
            normal: [texture_image_info(sampler, self.normal)],
            metallic_roughness: [texture_image_info(sampler, self.metallic_roughness)],
            occlusion: [texture_image_info(sampler, self.occlusion)],
            emissive: [texture_image_info(sampler, self.emissive)],
        }
    }
}

struct MaterialImageInfos {
    base_color: [vk::DescriptorImageInfo; 1],
    normal: [vk::DescriptorImageInfo; 1],
    metallic_roughness: [vk::DescriptorImageInfo; 1],
    occlusion: [vk::DescriptorImageInfo; 1],
    emissive: [vk::DescriptorImageInfo; 1],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MaterialParams {
    base_color_factor: [f32; 4],
    emissive_occlusion: [f32; 4],
    pbr_alpha: [f32; 4],
    flags: [u32; 4],
}

impl MaterialParams {
    /// Converts protocol material facts into the stable shader parameter layout.
    fn from_descriptor(
        descriptor: &MaterialDescriptor,
        texture_flags: MaterialTextureFlags,
    ) -> Self {
        let emissive = descriptor.emissive_factor();
        Self {
            base_color_factor: descriptor.base_color_factor(),
            emissive_occlusion: [
                emissive[0],
                emissive[1],
                emissive[2],
                descriptor.occlusion_strength_milli() as f32 / 1000.0,
            ],
            pbr_alpha: [
                descriptor.metallic_factor_milli() as f32 / 1000.0,
                descriptor.roughness_factor_milli() as f32 / 1000.0,
                descriptor.alpha_cutoff_milli() as f32 / 1000.0,
                descriptor.normal_scale_milli() as f32 / 1000.0,
            ],
            flags: [
                alpha_mode_code(descriptor.alpha_mode()),
                texture_flags.bits(),
                u32::from(descriptor.double_sided()),
                0,
            ],
        }
    }
}

/// Converts alpha modes into stable shader-side integer tags.
fn alpha_mode_code(mode: MaterialAlphaMode) -> u32 {
    match mode {
        MaterialAlphaMode::Opaque => 0,
        MaterialAlphaMode::Cutout => 1,
        MaterialAlphaMode::Transparent => 2,
    }
}

/// Creates the descriptor layout shared by uploaded material records.
fn create_material_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let mut bindings = vec![
        vk::DescriptorSetLayoutBinding::default()
            .binding(shader_interface::MATERIAL_PARAMS_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    for slot in [
        MaterialTextureSlot::BaseColor,
        MaterialTextureSlot::Normal,
        MaterialTextureSlot::MetallicRoughness,
        MaterialTextureSlot::Occlusion,
        MaterialTextureSlot::Emissive,
    ] {
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(shader_interface::material_texture_binding(slot))
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
    }
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    // Safety: the binding slice lives for the duration of the call.
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the bounded descriptor pool used for imported material records.
fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MATERIAL_DESCRIPTOR_CAPACITY),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MATERIAL_DESCRIPTOR_CAPACITY * 5),
    ];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(MATERIAL_DESCRIPTOR_CAPACITY)
        .pool_sizes(&pool_sizes);

    // Safety: pool sizes are local values and no custom allocation callbacks are used.
    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Creates the sampler shared by imported material texture descriptors.
fn create_sampler(device: &Device) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .max_lod(1.0);

    // Safety: sampler creation uses only local scalar values.
    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

/// Allocates one descriptor set for a material record.
fn allocate_material_descriptor_set(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    material_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, VulkanError> {
    let layouts = [material_set_layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    // Safety: the descriptor pool and set layout are alive for this allocation.
    unsafe { device.allocate_descriptor_sets(&allocate_info) }
        .map(|mut sets| sets.remove(0))
        .map_err(VulkanError::Vk)
}

/// Writes parameter and explicit texture descriptors for one material set.
fn update_material_descriptor_set(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    params: &GpuBuffer,
    sampler: vk::Sampler,
    textures: Option<MaterialTextureSet<'_>>,
) {
    let buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(params.handle())
        .offset(0)
        .range(size_of::<MaterialParams>() as vk::DeviceSize)];
    let image_infos = textures.map(|textures| textures.image_infos(sampler));
    let mut writes = vec![
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(shader_interface::MATERIAL_PARAMS_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info),
    ];
    if let Some(image_infos) = image_infos.as_ref() {
        writes.extend([
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(shader_interface::MATERIAL_BASE_COLOR_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos.base_color),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(shader_interface::MATERIAL_NORMAL_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos.normal),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(shader_interface::MATERIAL_METALLIC_ROUGHNESS_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos.metallic_roughness),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(shader_interface::MATERIAL_OCCLUSION_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos.occlusion),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(shader_interface::MATERIAL_EMISSIVE_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos.emissive),
        ]);
    }

    // Safety: descriptor set and resources are alive and the write slices outlive the call.
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

/// Builds one sampled-image descriptor for a texture that is already in shader-read layout.
fn texture_image_info(sampler: vk::Sampler, texture: &VulkanTexture) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(texture.image_view())
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
}

/// Destroys one Vulkan sampler.
fn destroy_sampler(device: &Device, sampler: vk::Sampler) {
    if sampler != vk::Sampler::null() {
        // Safety: the sampler was created by this device and is destroyed after descriptor use.
        unsafe {
            device.destroy_sampler(sampler, None);
        }
    }
}

/// Destroys one descriptor pool and all sets allocated from it.
fn destroy_descriptor_pool(device: &Device, pool: vk::DescriptorPool) {
    if pool != vk::DescriptorPool::null() {
        // Safety: the pool is destroyed after all descriptor users are idle.
        unsafe {
            device.destroy_descriptor_pool(pool, None);
        }
    }
}

/// Destroys one descriptor set layout.
fn destroy_descriptor_set_layout(device: &Device, layout: vk::DescriptorSetLayout) {
    if layout != vk::DescriptorSetLayout::null() {
        // Safety: descriptor set layouts are destroyed after dependent pools and pipelines.
        unsafe {
            device.destroy_descriptor_set_layout(layout, None);
        }
    }
}
