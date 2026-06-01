use std::{ffi::CString, fmt, io, io::Cursor, mem, path::PathBuf};

use ash::vk;

mod shader;

pub use shader::{HotReload, ShaderCode, ShaderSet};

#[derive(Debug)]
pub enum PipelineError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    InvalidSpv {
        shader: &'static str,
        source: io::Error,
    },
    NoShaderStages,
    Vk {
        op: &'static str,
        result: vk::Result,
    },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::InvalidSpv { shader, source } => write!(f, "{shader}: invalid SPIR-V: {source}"),
            Self::NoShaderStages => write!(f, "pipeline needs at least one shader stage"),
            Self::Vk { op, result } => write!(f, "{op}: {result:?}"),
        }
    }
}

impl std::error::Error for PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } | Self::InvalidSpv { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VertexLayout {
    pub bindings: Vec<vk::VertexInputBindingDescription>,
    pub attributes: Vec<vk::VertexInputAttributeDescription>,
}

impl VertexLayout {
    pub fn interleaved(stride: u32, attributes: &[VertexAttribute]) -> Self {
        Self {
            bindings: vec![vk::VertexInputBindingDescription {
                binding: 0,
                stride,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
            attributes: attributes
                .iter()
                .map(|attr| vk::VertexInputAttributeDescription {
                    binding: 0,
                    location: attr.location,
                    format: attr.format,
                    offset: attr.offset,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: vk::Format,
    pub offset: u32,
}

impl VertexAttribute {
    pub const fn new(location: u32, format: vk::Format, offset: u32) -> Self {
        Self {
            location,
            format,
            offset,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl ModelVertex {
    pub fn layout() -> VertexLayout {
        let normal = mem::size_of::<[f32; 3]>() as u32;
        let uv = normal * 2;

        VertexLayout::interleaved(
            mem::size_of::<Self>() as u32,
            &[
                VertexAttribute::new(0, vk::Format::R32G32B32_SFLOAT, 0),
                VertexAttribute::new(1, vk::Format::R32G32B32_SFLOAT, normal),
                VertexAttribute::new(2, vk::Format::R32G32_SFLOAT, uv),
            ],
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RasterizationConfig {
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub line_width: f32,
    pub depth_bias_constant: f32,
    pub depth_bias_slope: f32,
}

impl RasterizationConfig {
    pub fn shadow() -> Self {
        Self {
            cull_mode: vk::CullModeFlags::BACK,
            ..Self::default()
        }
    }

    pub fn double_sided() -> Self {
        Self {
            cull_mode: vk::CullModeFlags::NONE,
            ..Self::default()
        }
    }
}

impl Default for RasterizationConfig {
    fn default() -> Self {
        Self {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            depth_bias_constant: 0.0,
            depth_bias_slope: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ColorBlendConfig {
    pub enabled: bool,
    pub src_color: vk::BlendFactor,
    pub dst_color: vk::BlendFactor,
    pub color_op: vk::BlendOp,
    pub src_alpha: vk::BlendFactor,
    pub dst_alpha: vk::BlendFactor,
    pub alpha_op: vk::BlendOp,
}

impl ColorBlendConfig {
    pub fn alpha() -> Self {
        Self {
            enabled: true,
            src_color: vk::BlendFactor::SRC_ALPHA,
            dst_color: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_op: vk::BlendOp::ADD,
            src_alpha: vk::BlendFactor::ONE,
            dst_alpha: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_op: vk::BlendOp::ADD,
        }
    }

    pub fn additive() -> Self {
        Self {
            enabled: true,
            src_color: vk::BlendFactor::ONE,
            dst_color: vk::BlendFactor::ONE,
            color_op: vk::BlendOp::ADD,
            src_alpha: vk::BlendFactor::ONE,
            dst_alpha: vk::BlendFactor::ONE,
            alpha_op: vk::BlendOp::ADD,
        }
    }
}

impl Default for ColorBlendConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            src_color: vk::BlendFactor::ONE,
            dst_color: vk::BlendFactor::ZERO,
            color_op: vk::BlendOp::ADD,
            src_alpha: vk::BlendFactor::ONE,
            dst_alpha: vk::BlendFactor::ZERO,
            alpha_op: vk::BlendOp::ADD,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DepthConfig {
    pub format: vk::Format,
    pub write: bool,
    pub compare: vk::CompareOp,
}

impl Default for DepthConfig {
    fn default() -> Self {
        Self {
            format: vk::Format::D32_SFLOAT,
            write: true,
            compare: vk::CompareOp::LESS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineDesc {
    pub shaders: ShaderSet,
    pub vertex_layout: VertexLayout,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub push_constants: Vec<vk::PushConstantRange>,
    pub topology: vk::PrimitiveTopology,
    pub rasterization: RasterizationConfig,
    pub color_blend: ColorBlendConfig,
    pub color_attachment: bool,
    pub depth: Option<DepthConfig>,
}

impl PipelineDesc {
    pub fn new(shaders: ShaderSet) -> Self {
        Self {
            shaders,
            vertex_layout: VertexLayout::default(),
            set_layouts: Vec::new(),
            push_constants: Vec::new(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            rasterization: RasterizationConfig::default(),
            color_blend: ColorBlendConfig::default(),
            color_attachment: true,
            depth: None,
        }
    }

    pub fn with_vertex_layout(mut self, vertex_layout: VertexLayout) -> Self {
        self.vertex_layout = vertex_layout;
        self
    }

    pub fn with_layout(
        mut self,
        set_layouts: Vec<vk::DescriptorSetLayout>,
        push_constants: Vec<vk::PushConstantRange>,
    ) -> Self {
        self.set_layouts = set_layouts;
        self.push_constants = push_constants;
        self
    }

    pub fn with_depth(mut self, depth: DepthConfig) -> Self {
        self.depth = Some(depth);
        self
    }

    pub fn with_color_blend(mut self, color_blend: ColorBlendConfig) -> Self {
        self.color_blend = color_blend;
        self
    }

    pub fn with_rasterization(mut self, rasterization: RasterizationConfig) -> Self {
        self.rasterization = rasterization;
        self
    }

    pub fn without_color_attachment(mut self) -> Self {
        self.color_attachment = false;
        self
    }

    pub fn build(
        &self,
        device: &ash::Device,
        cache: vk::PipelineCache,
        color_format: vk::Format,
    ) -> Result<GraphicsPipeline, PipelineError> {
        GraphicsPipeline::create(device, cache, color_format, self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GraphicsPipeline {
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
}

impl GraphicsPipeline {
    fn create(
        device: &ash::Device,
        cache: vk::PipelineCache,
        color_format: vk::Format,
        desc: &PipelineDesc,
    ) -> Result<Self, PipelineError> {
        if desc.shaders.stages.is_empty() {
            return Err(PipelineError::NoShaderStages);
        }

        let mut modules = Vec::with_capacity(desc.shaders.stages.len());
        let mut entry_names = Vec::with_capacity(desc.shaders.stages.len());

        for stage in &desc.shaders.stages {
            let bytes = stage.code.load()?;
            modules.push(ShaderModule::new(device, stage.name, &bytes)?);
            entry_names.push(CString::new(stage.entry).unwrap());
        }

        let stages: Vec<_> = desc
            .shaders
            .stages
            .iter()
            .zip(&modules)
            .zip(&entry_names)
            .map(|((stage, module), entry)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(stage.stage)
                    .module(module.handle)
                    .name(entry)
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&desc.vertex_layout.bindings)
            .vertex_attribute_descriptions(&desc.vertex_layout.attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(desc.topology)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(desc.rasterization.polygon_mode)
            .cull_mode(desc.rasterization.cull_mode)
            .front_face(desc.rasterization.front_face)
            .line_width(desc.rasterization.line_width)
            .depth_bias_enable(
                desc.rasterization.depth_bias_constant != 0.0
                    || desc.rasterization.depth_bias_slope != 0.0,
            )
            .depth_bias_constant_factor(desc.rasterization.depth_bias_constant)
            .depth_bias_slope_factor(desc.rasterization.depth_bias_slope);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(desc.color_blend.enabled)
            .src_color_blend_factor(desc.color_blend.src_color)
            .dst_color_blend_factor(desc.color_blend.dst_color)
            .color_blend_op(desc.color_blend.color_op)
            .src_alpha_blend_factor(desc.color_blend.src_alpha)
            .dst_alpha_blend_factor(desc.color_blend.dst_alpha)
            .alpha_blend_op(desc.color_blend.alpha_op);

        let color_blend_attachments = desc
            .color_attachment
            .then_some(color_blend_attachment)
            .into_iter()
            .collect::<Vec<_>>();

        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&desc.set_layouts)
            .push_constant_ranges(&desc.push_constants);

        let layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|result| PipelineError::Vk {
                    op: "create pipeline layout",
                    result,
                })?
        };

        let color_formats = [color_format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default();
        if desc.color_attachment {
            rendering = rendering.color_attachment_formats(&color_formats);
        }
        if let Some(depth) = desc.depth {
            rendering = rendering.depth_attachment_format(depth.format);
        }

        let depth_state = desc.depth.map(|depth| {
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(depth.write)
                .depth_compare_op(depth.compare)
        });

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(vk::RenderPass::null())
            .subpass(0)
            .push_next(&mut rendering);

        if let Some(depth_state) = &depth_state {
            pipeline_info = pipeline_info.depth_stencil_state(depth_state);
        }

        let pipelines =
            match unsafe { device.create_graphics_pipelines(cache, &[pipeline_info], None) } {
                Ok(pipelines) => pipelines,
                Err((partial, result)) => {
                    for pipeline in partial {
                        unsafe { device.destroy_pipeline(pipeline, None) };
                    }
                    unsafe { device.destroy_pipeline_layout(layout, None) };
                    return Err(PipelineError::Vk {
                        op: "create graphics pipeline",
                        result,
                    });
                }
            };

        Ok(Self {
            layout,
            handle: pipelines[0],
        })
    }

    /// # Safety
    ///
    /// The pipeline must not be in use by any in-flight command buffer.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.handle != vk::Pipeline::null() {
            unsafe { device.destroy_pipeline(self.handle, None) };
            self.handle = vk::Pipeline::null();
        }

        if self.layout != vk::PipelineLayout::null() {
            unsafe { device.destroy_pipeline_layout(self.layout, None) };
            self.layout = vk::PipelineLayout::null();
        }
    }
}

struct ShaderModule<'a> {
    device: &'a ash::Device,
    handle: vk::ShaderModule,
}

impl<'a> ShaderModule<'a> {
    fn new(
        device: &'a ash::Device,
        shader: &'static str,
        bytes: &[u8],
    ) -> Result<Self, PipelineError> {
        let mut cursor = Cursor::new(bytes);
        let code = ash::util::read_spv(&mut cursor)
            .map_err(|source| PipelineError::InvalidSpv { shader, source })?;

        let info = vk::ShaderModuleCreateInfo::default().code(&code);
        let handle = unsafe {
            device
                .create_shader_module(&info, None)
                .map_err(|result| PipelineError::Vk {
                    op: "create shader module",
                    result,
                })?
        };

        Ok(Self { device, handle })
    }
}

impl Drop for ShaderModule<'_> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_shader_module(self.handle, None) };
    }
}

pub fn create_pipeline_cache(device: &ash::Device) -> vk::PipelineCache {
    let info = vk::PipelineCacheCreateInfo::default();

    unsafe {
        device
            .create_pipeline_cache(&info, None)
            .expect("renderer init: failed to create pipeline cache")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rasterization_culls_back_faces() {
        let rasterization = RasterizationConfig::default();

        assert_eq!(rasterization.cull_mode, vk::CullModeFlags::BACK);
        assert_eq!(rasterization.front_face, vk::FrontFace::COUNTER_CLOCKWISE);
    }
}
