use crate::{import::ImportedTexture, protocol::TextureDescriptor};

#[derive(Clone, Debug)]
pub(crate) struct GpuTextureAsset {
    descriptor: TextureDescriptor,
}

impl GpuTextureAsset {
    /// Stores an imported texture payload until the Vulkan texture uploader consumes it.
    pub(crate) fn from_imported(imported: &ImportedTexture) -> Self {
        let descriptor = imported.descriptor().clone();
        tracing::trace!(
            width = descriptor.width(),
            height = descriptor.height(),
            format = descriptor.format().name(),
            bytes = descriptor.pixels().len(),
            "registered texture payload"
        );

        Self { descriptor }
    }

    /// Returns whether this texture has a non-empty validated pixel payload.
    pub(crate) fn has_pixels(&self) -> bool {
        self.descriptor.width() > 0
            && self.descriptor.height() > 0
            && !self.descriptor.pixels().is_empty()
    }

    /// Returns the immutable texture descriptor consumed by backend image upload.
    pub(crate) fn descriptor(&self) -> &TextureDescriptor {
        &self.descriptor
    }
}
