use ash::{Device, vk};

use crate::protocol::NonZeroExtent;

/// Creates the color/depth render pass used by the graph scene pass.
pub(super) fn create_scene_render_pass(
    device: &Device,
    color_format: vk::Format,
    normal_roughness_format: vk::Format,
    transparent_normal_roughness_format: vk::Format,
    depth_format: vk::Format,
) -> Result<vk::RenderPass, super::VulkanError> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let normal_roughness_attachment = vk::AttachmentDescription::default()
        .format(normal_roughness_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let transparent_normal_roughness_attachment = vk::AttachmentDescription::default()
        .format(transparent_normal_roughness_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let normal_roughness_attachment_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let transparent_normal_roughness_attachment_ref = vk::AttachmentReference::default()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_attachment_ref = vk::AttachmentReference::default()
        .attachment(3)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_attachment_refs = [
        color_attachment_ref,
        normal_roughness_attachment_ref,
        transparent_normal_roughness_attachment_ref,
    ];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref);
    let attachments = [
        color_attachment,
        normal_roughness_attachment,
        transparent_normal_roughness_attachment,
        depth_attachment,
    ];
    let subpasses = [subpass];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);

    // Safety: all slices in `create_info` live for the duration of the call, and graph barriers
    // transition the swapchain image into and out of the color attachment layout.
    unsafe { device.create_render_pass(&create_info, None) }.map_err(super::VulkanError::Vk)
}

/// Creates the render pass that writes moment shadow data while depth-testing casters.
pub(super) fn create_shadow_render_pass(
    device: &Device,
    moment_format: vk::Format,
    depth_format: vk::Format,
) -> Result<vk::RenderPass, super::VulkanError> {
    let moment_attachment = vk::AttachmentDescription::default()
        .format(moment_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let moment_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_attachment_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_attachment_refs = [moment_attachment_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref);
    let attachments = [moment_attachment, depth_attachment];
    let subpasses = [subpass];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);

    // Safety: all create-info slices live for this call and graph barriers control moment layout.
    unsafe { device.create_render_pass(&create_info, None) }.map_err(super::VulkanError::Vk)
}

/// Creates the color-only pass that accumulates transparent shadow transmittance per cascade.
pub(super) fn create_translucent_shadow_render_pass(
    device: &Device,
    color_format: vk::Format,
) -> Result<vk::RenderPass, super::VulkanError> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_attachment_refs = [color_attachment_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs);
    let attachments = [color_attachment];
    let subpasses = [subpass];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);

    // Safety: graph barriers place the transmittance target in color attachment layout.
    unsafe { device.create_render_pass(&create_info, None) }.map_err(super::VulkanError::Vk)
}

/// Creates the color-only render pass that writes a post result into the swapchain image.
pub(super) fn create_post_render_pass(
    device: &Device,
    format: vk::Format,
) -> Result<vk::RenderPass, super::VulkanError> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_attachment_refs = [color_attachment_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs);
    let attachments = [color_attachment];
    let subpasses = [subpass];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);

    // Safety: all create-info slices are local and graph barriers handle image layout changes.
    unsafe { device.create_render_pass(&create_info, None) }.map_err(super::VulkanError::Vk)
}

/// Creates the color-only pass used by separable moment shadow blur targets.
pub(super) fn create_shadow_blur_render_pass(
    device: &Device,
    moment_format: vk::Format,
) -> Result<vk::RenderPass, super::VulkanError> {
    create_post_render_pass(device, moment_format)
}

/// Destroys one swapchain render pass after its framebuffers are gone.
pub(super) fn destroy_render_pass(device: &Device, render_pass: vk::RenderPass) {
    if render_pass == vk::RenderPass::null() {
        return;
    }

    // Safety: the render pass was created by this device and is destroyed after its framebuffers.
    unsafe {
        device.destroy_render_pass(render_pass, None);
    }
}

/// Creates the scene framebuffer that binds scene color and scene depth views.
pub(super) fn create_scene_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    color_view: vk::ImageView,
    normal_roughness_view: vk::ImageView,
    transparent_normal_roughness_view: vk::ImageView,
    depth_view: vk::ImageView,
    extent: NonZeroExtent,
) -> Result<vk::Framebuffer, super::VulkanError> {
    let attachments = [
        color_view,
        normal_roughness_view,
        transparent_normal_roughness_view,
        depth_view,
    ];
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(extent.width())
        .height(extent.height())
        .layers(1);

    // Safety: both image views match the render pass attachments and outlive the framebuffer.
    unsafe { device.create_framebuffer(&create_info, None) }.map_err(super::VulkanError::Vk)
}

/// Creates the shadow framebuffer that binds moment color and private depth views.
pub(super) fn create_shadow_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    moment_view: vk::ImageView,
    depth_view: vk::ImageView,
    extent: NonZeroExtent,
) -> Result<vk::Framebuffer, super::VulkanError> {
    let attachments = [moment_view, depth_view];
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(extent.width())
        .height(extent.height())
        .layers(1);

    // Safety: both image views match the shadow render pass attachments.
    unsafe { device.create_framebuffer(&create_info, None) }.map_err(super::VulkanError::Vk)
}

/// Creates a framebuffer for one translucent shadow cascade transmittance target.
pub(super) fn create_translucent_shadow_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    color_view: vk::ImageView,
    extent: NonZeroExtent,
) -> Result<vk::Framebuffer, super::VulkanError> {
    let attachments = [color_view];
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(extent.width())
        .height(extent.height())
        .layers(1);

    // Safety: the image view belongs to the cascade and matches the render pass attachment.
    unsafe { device.create_framebuffer(&create_info, None) }.map_err(super::VulkanError::Vk)
}

/// Creates one post framebuffer for a swapchain image view.
pub(super) fn create_post_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    image_view: vk::ImageView,
    extent: NonZeroExtent,
) -> Result<vk::Framebuffer, super::VulkanError> {
    let attachments = [image_view];
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(extent.width())
        .height(extent.height())
        .layers(1);

    // Safety: the image view matches the color-only post render pass.
    unsafe { device.create_framebuffer(&create_info, None) }.map_err(super::VulkanError::Vk)
}

/// Destroys one framebuffer created for a swapchain image view.
pub(super) fn destroy_framebuffer(device: &Device, framebuffer: vk::Framebuffer) {
    if framebuffer == vk::Framebuffer::null() {
        return;
    }

    // Safety: the framebuffer was created by this device and is destroyed exactly once.
    unsafe {
        device.destroy_framebuffer(framebuffer, None);
    }
}
