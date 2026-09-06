use std::{ffi::CStr, mem::size_of};

use ash::{Device, vk};

use crate::{
    protocol::{CameraSnapshot, FrameSnapshot, LightPacket, NonZeroExtent, RenderQualitySettings},
    renderer::{
        graph::{
            FrameGraphPlan, GraphResource, ResourceState, TAA_DEPTH_HISTORY_RESOURCES,
            TAA_HISTORY_COUNT, TAA_HISTORY_RESOURCES, TAA_MOTION_RESOURCE,
            TAA_NORMAL_HISTORY_RESOURCES, TAA_RESOLVE_PASS,
        },
        pipeline::shader_interface,
    },
};

use super::{
    VulkanError,
    buffer::{GpuBuffer, create_host_buffer, write_buffer_value},
    shader::{self, assets},
    swapchain_target::{ColorTarget, create_color_target, destroy_color_target},
};

pub(super) const TAA_HISTORY_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub(super) const TAA_MOTION_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;
pub(super) const TAA_LINEAR_DEPTH_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
pub(super) const TAA_NORMAL_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// SMAA is the production path for now; keep the temporal implementation available to graph
/// tests/tools without allocating its full-resolution resources on every swapchain.
pub(super) const TEMPORAL_AA_ENABLED: bool = false;

const HISTORY_SAMPLE_LIMIT: f32 = 16.0;
// Volumetric lighting is a deterministic screen-space field, but its CSM/PCSS visibility is still
// reconstructed from a finite XY lattice.  Let the final full-resolution TAA integrate a longer
// sequence than opaque color so one lattice phase cannot remain visible as a pulsing box.
const VOLUMETRIC_HISTORY_SAMPLE_LIMIT: f32 = 32.0;
// The profile feedback values below are calibrated at the default window pacing rate.  Doubling
// this time scale intentionally slows convergence while preserving the same half-life in seconds
// when the configured presentation rate changes.
const TAA_REFERENCE_FRAME_RATE_HZ: f32 = 120.0;
const TAA_CONVERGENCE_TIME_SCALE: f32 = 2.0;
const TAA_LN_TWO: f32 = std::f32::consts::LN_2;
// At the reference pacing this is two complete Halton camera-jitter cycles.  Expressing the floor
// in seconds keeps the temporal response frame-rate invariant while still making the configured
// 120 Hz path slower than one jitter cycle.
const TAA_MIN_JITTER_HALF_LIFE_CYCLES: f32 = 2.0;
const TAA_MIN_HALF_LIFE_SECONDS: f32 =
    JITTER_SAMPLE_COUNT as f32 * TAA_MIN_JITTER_HALF_LIFE_CYCLES / TAA_REFERENCE_FRAME_RATE_HZ;
// A directional shadow field moves when the Sun moves even if the camera and every opaque
// surface stay still.  Keep this response separate from the camera reprojection thresholds:
// sub-pixel light motion should shorten the volumetric history smoothly, while a light cut should
// still be handled by the existing signature reset.
// CSM projection bases are rebuilt from the moving sun direction every frame.  A 2.5 mrad
// threshold leaves most of that motion hidden behind a saturated 32-sample history, so the
// shadowed GodRay silhouette trails behind the sun as repeated boxes.  React within 0.5 mrad:
// this still leaves static lights fully temporally stable, but gives a moving directional light
// enough current-frame weight to follow the shadow field instead of ghosting it.
const VOLUMETRIC_LIGHT_ANGLE_FULL_RESPONSE: f32 = 0.0005;
const VOLUMETRIC_LIGHT_INTENSITY_FULL_RESPONSE: f32 = 0.02;
const VOLUMETRIC_LIGHT_COLOR_FULL_RESPONSE: f32 = 0.02;
// PCSS history is reprojected with the camera matrix. A small view rotation/translation can move
// a shadow edge by several pixels even when the Sun is static, so keep a separate camera-motion
// response signal instead of letting the long PCSS history treat that edge as stationary.
const PCSS_CAMERA_ANGLE_FULL_RESPONSE_RADIANS: f32 = 0.008726646;
const PCSS_CAMERA_TRANSLATION_FULL_RESPONSE_FRACTION: f32 = 0.01;
// A moving Sun changes the shadow field, but ordinary motion must still benefit from temporal
// averaging. Keep the light component below the camera component in the packed PCSS reactivity
// lane; only the explicit light-cut predicate below discards the history completely.
const PCSS_LIGHT_REACTIVITY_SCALE: f32 = 0.5;
// A genuinely replaced Sun invalidates the old shadow/scattering field.  This threshold is kept
// in physical light units rather than using `volumetric_light_change`: that signal is deliberately
// sensitive to sub-milliradian motion and must never reset the history every frame during a normal
// day/night orbit.
const TAA_LIGHT_CUT_ANGLE_RADIANS: f32 = 0.35;
const TAA_LIGHT_CUT_INTENSITY_RATIO: f32 = 3.0;
const TAA_LIGHT_CUT_COLOR_DELTA: f32 = 1.5;
const DEPTH_REJECT_ABSOLUTE: f32 = 0.01;
const DEPTH_REJECT_RELATIVE: f32 = 0.005;
const NORMAL_REJECT_COSINE: f32 = 0.90;
const JITTER_SAMPLE_COUNT: u64 = 16;
// Keep the volumetric/PCSS sample phase independent from the 16-sample camera jitter cycle. The
// effect histories retain roughly 15 frames at the reference rate; repeating the same shadow/volume
// estimate after exactly one jitter cycle would leave a visible periodic layer before it has
// decayed. A 64-frame Van der Corput horizon keeps every retained contribution decorrelated.
const TEMPORAL_PHASE_SAMPLE_COUNT: u64 = 64;
const JITTER_SEQUENCE_CENTROID: [f32; 2] = [-0.029_296_875, -0.037_037_037];
const CURRENT_COLOR_INPUT_COUNT: usize = 2;
const SCENE_COLOR_INPUT_INDEX: usize = 0;
const CORRECTED_SCENE_COLOR_INPUT_INDEX: usize = 1;

const CURRENT_COLOR_BINDING: u32 = 0;
const CURRENT_DEPTH_BINDING: u32 = 1;
const CURRENT_NORMAL_BINDING: u32 = 2;
const PREVIOUS_COLOR_BINDING: u32 = 3;
const PREVIOUS_DEPTH_BINDING: u32 = 4;
const PREVIOUS_NORMAL_BINDING: u32 = 5;
const PARAMS_BINDING: u32 = 6;
const CURRENT_TRANSPARENT_NORMAL_BINDING: u32 = 7;

const VERTEX_SHADER: &[u8] = assets::POST_VERT;
const FRAGMENT_SHADER: &[u8] = assets::POST_TAA_RESOLVE_FRAG;
const SHADER_ENTRY: &CStr = shader::ENTRY;

#[repr(C)]
#[derive(Clone, Copy)]
struct TemporalResolveUniform {
    current_view_projection: [f32; 16],
    inverse_current_view_projection: [f32; 16],
    previous_view_projection: [f32; 16],
    inverse_current_view: [f32; 16],
    inverse_previous_view: [f32; 16],
    texel_feedback_reset: [f32; 4],
    rejection: [f32; 4],
    jitter_pixels: [f32; 4],
    /// x = current color is the volumetric resolve, y = its longer history limit.
    /// Volumetric lighting is spatially reconstructed each frame, so its history needs a separate
    /// accumulation horizon and current-color floor from ordinary opaque scene color.
    effects: [f32; 4],
}

struct TemporalHistoryTarget {
    color: ColorTarget,
    linear_depth: ColorTarget,
    normal: ColorTarget,
    color_state: ResourceState,
    depth_state: ResourceState,
    normal_state: ResourceState,
}

struct TemporalMotionTarget {
    color: ColorTarget,
    state: ResourceState,
}

#[derive(Clone, Copy)]
struct PreviousFrame {
    frame_id: u64,
    scene: u64,
    surface_generation: u64,
    camera: CameraSnapshot,
    aspect: f32,
    view_projection: [f32; 16],
    jitter_pixels: [f32; 2],
    aa_blend: f32,
    /// Volumetric GodRay input has its own temporal contract even when ordinary color TAA is
    /// disabled. Keep this bit in the reprojection key so switching between scene and volume
    /// color cannot reuse an incompatible history.
    volumetric_input: bool,
    /// Hash of the camera-ray medium and directional-light state used by volumetric lighting.
    /// Volumetric history is invalid whenever this changes, even if the opaque scene is unchanged.
    volumetric_signature: u64,
    /// Raw directional-light state for the continuous volumetric history response. The signature
    /// above is intentionally quantized, but the TAA blend must still react between signature
    /// epochs or a moving CSM silhouette is averaged into a stack of ghost boxes.
    light_direction: [f32; 3],
    light_intensity: f32,
    light_color: [f32; 3],
}

#[derive(Clone, Copy)]
struct PendingFrame {
    previous: PreviousFrame,
    slot_index: usize,
    write_history_index: usize,
    current_color_input_index: usize,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(super) struct TaaFrameInfo {
    pub(super) jittered_view_projection: [f32; 16],
    pub(super) inverse_current_view_projection: [f32; 16],
    pub(super) previous_view_projection: [f32; 16],
    pub(super) inverse_current_view: [f32; 16],
    pub(super) inverse_previous_view: [f32; 16],
    pub(super) current_jitter_pixels: [f32; 2],
    pub(super) previous_jitter_pixels: [f32; 2],
    pub(super) write_history_index: usize,
    /// Resets only consumers of the shared camera/depth/normal history (camera cut, scene change,
    /// resize, or stale frame). This deliberately ignores whether color TAA itself is enabled.
    pub(super) reset_reprojection_history: bool,
}

/// Camera/light reprojection metadata shared by temporal volumetric and shadow consumers.
///
/// Ordinary color TAA remains optional, but these two effects still need one authoritative
/// discontinuity decision and the exact previous camera matrix. Keeping that state here avoids
/// each effect inventing a subtly different reset rule.
#[derive(Clone, Copy)]
pub(super) struct TemporalEffectFrame {
    pub(super) previous_view_projection: [f32; 16],
    /// Frame-rate-aware recursive feedback for the Sun Shaft/volume history.
    pub(super) sun_feedback: f32,
    /// Frame-rate-aware recursive feedback for the PCSS visibility history.
    pub(super) pcss_feedback: f32,
    /// Shared low-discrepancy phase used to decorrelate volumetric quadrature and PCSS taps
    /// between frames. Temporal accumulation is ineffective when every frame evaluates the same
    /// deterministic approximation, because the approximation error becomes a stable layer.
    pub(super) sample_phase: f32,
    /// Bounded directional-light motion signal shared by the volumetric and PCSS history shaders.
    /// This occupies an existing temporal-uniform lane, so the descriptor ABI stays unchanged.
    pub(super) light_motion: f32,
    /// Camera motion response used only by the PCSS visibility history. The volume keeps the light
    /// signal separate because its endpoint reprojection already accounts for camera movement.
    pub(super) pcss_camera_motion: f32,
    /// True when the shared camera/light contract cannot reuse either history.
    pub(super) reset: bool,
}

impl Default for TemporalEffectFrame {
    fn default() -> Self {
        Self {
            previous_view_projection: identity_mat4(),
            sun_feedback: 0.0,
            pcss_feedback: 0.0,
            sample_phase: 0.0,
            light_motion: 0.0,
            pcss_camera_motion: 0.0,
            reset: true,
        }
    }
}

impl TemporalEffectFrame {
    /// Returns the PCSS history reactivity while preserving useful accumulation during ordinary
    /// directional-light motion. Camera movement remains fully reactive because its reprojection
    /// can move a receiver across a screen-space shadow edge immediately.
    pub(super) fn pcss_reactivity(self) -> f32 {
        self.pcss_camera_motion
            .max(self.light_motion * PCSS_LIGHT_REACTIVITY_SCALE)
            .clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy)]
struct TemporalEffectPrevious {
    frame_id: u64,
    scene: u64,
    surface_generation: u64,
    camera: CameraSnapshot,
    aspect: f32,
    view_projection: [f32; 16],
    light_direction: [f32; 3],
    light_intensity: f32,
    light_color: [f32; 3],
}

#[derive(Default)]
pub(super) struct TemporalEffectsState {
    previous: Option<TemporalEffectPrevious>,
    pending: Option<TemporalEffectPrevious>,
}

impl TemporalEffectsState {
    pub(super) fn prepare(
        &mut self,
        snapshot: &FrameSnapshot,
        camera: CameraSnapshot,
        extent: vk::Extent2D,
        light: LightPacket,
    ) -> TemporalEffectFrame {
        let aspect = extent.width.max(1) as f32 / extent.height.max(1) as f32;
        let view_projection = camera.view_projection(aspect);
        let light_direction = normalize3(light.direction);
        let previous = self.previous;
        let reset = previous.is_none_or(|previous| {
            temporal_effect_discontinuous(previous, snapshot, camera, aspect, light)
        });
        let light_motion = previous.map_or(0.0, |previous| {
            temporal_effect_light_change(light, previous)
        });
        let pcss_camera_motion = previous.map_or(0.0, |previous| {
            pcss_camera_motion_signal(camera, previous.camera)
        });
        let previous_view_projection =
            previous.map_or(view_projection, |previous| previous.view_projection);
        let frame_rate_hz = snapshot.frame_rate_hz;
        let frame = TemporalEffectFrame {
            previous_view_projection,
            sun_feedback: if reset {
                0.0
            } else {
                temporal_effect_feedback(frame_rate_hz, 0.955)
            },
            pcss_feedback: if reset {
                0.0
            } else {
                temporal_effect_feedback(frame_rate_hz, 0.86)
            },
            sample_phase: temporal_sample_phase(snapshot.frame_id.raw(), light, light_motion),
            light_motion,
            pcss_camera_motion,
            reset,
        };
        let pending = TemporalEffectPrevious {
            frame_id: snapshot.frame_id.raw(),
            scene: snapshot.scene.raw(),
            surface_generation: snapshot.surface_generation.raw(),
            camera,
            aspect,
            view_projection,
            light_direction,
            light_intensity: light.intensity,
            light_color: light.color,
        };
        self.pending = Some(pending);
        frame
    }

    pub(super) fn commit(&mut self) {
        if let Some(previous) = self.pending.take() {
            self.previous = Some(previous);
        }
    }

    pub(super) fn invalidate(&mut self) {
        self.previous = None;
        self.pending = None;
    }
}

fn temporal_effect_discontinuous(
    previous: TemporalEffectPrevious,
    snapshot: &FrameSnapshot,
    camera: CameraSnapshot,
    aspect: f32,
    light: LightPacket,
) -> bool {
    if snapshot.frame_id.raw() <= previous.frame_id
        || snapshot.scene.raw() != previous.scene
        || snapshot.surface_generation.raw() != previous.surface_generation
        || (aspect - previous.aspect).abs() > 0.0005
    {
        return true;
    }

    let eye_delta = distance3(camera.eye, previous.camera.eye);
    let view_distance = distance3(previous.camera.eye, previous.camera.target).max(1.0);
    let forward = normalize3(sub3(camera.target, camera.eye));
    let previous_forward = normalize3(sub3(previous.camera.target, previous.camera.eye));
    let direction_cosine = dot3(forward, previous_forward);
    let fov_delta = (camera.fov_y_radians - previous.camera.fov_y_radians).abs();
    let near_delta = (camera.near - previous.camera.near).abs() / previous.camera.near.max(0.0001);
    let far_delta = (camera.far - previous.camera.far).abs() / previous.camera.far.max(0.0001);
    if eye_delta > (view_distance * 2.0).max(5.0)
        || direction_cosine < 35.0_f32.to_radians().cos()
        || fov_delta > 8.0_f32.to_radians()
        || near_delta > 0.10
        || far_delta > 0.10
    {
        return true;
    }

    let previous_direction = normalize3(previous.light_direction);
    let direction_cross = cross3(light_direction(light), previous_direction);
    let direction_cosine = dot3(light_direction(light), previous_direction).clamp(-1.0, 1.0);
    if direction_cross
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .atan2(direction_cosine)
        .abs()
        > TAA_LIGHT_CUT_ANGLE_RADIANS
    {
        return true;
    }
    let current_intensity = light.intensity.max(0.0);
    let previous_intensity = previous.light_intensity.max(0.0);
    let intensity_max = current_intensity.max(previous_intensity);
    if intensity_max > 0.25
        && intensity_max / current_intensity.min(previous_intensity).max(0.01)
            > TAA_LIGHT_CUT_INTENSITY_RATIO
    {
        return true;
    }
    let color_delta = light
        .color
        .into_iter()
        .zip(previous.light_color)
        .map(|(current, old)| (current.max(0.0) - old.max(0.0)).abs() / current.max(old).max(0.05))
        .fold(0.0_f32, f32::max);
    color_delta > TAA_LIGHT_CUT_COLOR_DELTA
}

fn light_direction(light: LightPacket) -> [f32; 3] {
    normalize3(light.direction)
}

fn temporal_effect_feedback(frame_rate_hz: f32, reference_feedback: f32) -> f32 {
    let frame_rate = if frame_rate_hz.is_finite() {
        frame_rate_hz.clamp(15.0, 240.0)
    } else {
        TAA_REFERENCE_FRAME_RATE_HZ
    };
    // The profile value is defined as the retained history after one frame at the reference
    // pacing rate. Convert that frame count to seconds before taking a step at the actual rate;
    // omitting this conversion makes the feedback approach one and effectively freezes the
    // history as soon as the application runs below the reference rate.
    let half_life_frames = -TAA_LN_TWO / reference_feedback.clamp(0.001, 0.999).ln();
    let half_life_seconds = half_life_frames / TAA_REFERENCE_FRAME_RATE_HZ;
    (-TAA_LN_TWO / (half_life_seconds.max(1.0e-4) * frame_rate)).exp()
}

/// Returns a low-discrepancy temporal phase shared by volumetric quadrature and PCSS taps.
///
/// A stable spatial approximation cannot be improved by recursive history on its own: every
/// frame contributes the same layer error. A short base-3 sequence changes only the sampling
/// phase, while reprojection/history validation keeps the resulting estimate stable on screen.
/// The phase also follows the directional light. A frame-only sequence can change the nominal
/// phase while the moving Sun still reuses almost the same shadow lattice, leaving a layered shaft
/// in the history.
fn temporal_sample_phase(frame_id: u64, light: LightPacket, light_motion: f32) -> f32 {
    // Avoid the zero index so neither stratified half of a volume slice lands exactly on its
    // boundary. Keep this horizon longer than the camera jitter cycle so recursive history does
    // not see the same finite-slice error again while its previous contribution is still strong.
    let frame_phase = halton(frame_id % TEMPORAL_PHASE_SAMPLE_COUNT + 1, 3);
    let direction = normalize3(light.direction);
    // Incommensurate coefficients turn the light direction into a smooth phase anchor;
    // unlike a quantized hash, even sub-milliradian Sun motion changes the phase continuously.
    let direction_phase =
        (direction[0] * 17.0 + direction[1] * 31.0 + direction[2] * 47.0).rem_euclid(1.0);
    let intensity = if light.intensity.is_finite() {
        light.intensity.max(0.0)
    } else {
        0.0
    };
    let color_phase = light
        .color
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let value = if value.is_finite() {
                value.max(0.0)
            } else {
                0.0
            };
            value * [0.071, 0.113, 0.173][index]
        })
        .sum::<f32>()
        .rem_euclid(1.0);
    let light_phase =
        (direction_phase * 0.73 + color_phase * 0.19 + intensity * 0.013).rem_euclid(1.0);
    // The motion term is strong enough to decorrelate the volumetric PCSS lattice at the
    // sub-milliradian response threshold, while the frame sequence remains dominant for a static
    // light.
    (frame_phase + light_phase * 0.41 + light_motion.clamp(0.0, 1.0) * 0.23)
        .rem_euclid(1.0)
        .clamp(0.03125, 0.96875)
}

/// Owns the HDR TAA resolve plus the motion/depth/normal history shared by temporal effects.
pub(super) struct TemporalAntiAliasing {
    enabled: bool,
    histories: Vec<TemporalHistoryTarget>,
    motion: TemporalMotionTarget,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    color_sampler: vk::Sampler,
    data_sampler: vk::Sampler,
    uniform_buffers: Vec<GpuBuffer>,
    history_write_index: usize,
    history_valid: bool,
    /// Previous-frame depth/normal histories are safe for pre-TAA consumers only when the
    /// reprojection contract was continuous for the frame being prepared.  Legacy low-resolution
    /// GodRay mask generation runs before this frame's TAA pass, so it uses this flag to avoid
    /// sampling an undefined/stale stable depth image on the first frame or after a camera cut.
    stable_metadata_valid: bool,
    frame_index: u64,
    previous: Option<PreviousFrame>,
    pending: Option<PendingFrame>,
}

struct TemporalBuild<'a> {
    device: &'a Device,
    histories: Vec<TemporalHistoryTarget>,
    motion: Option<TemporalMotionTarget>,
    render_pass: Option<vk::RenderPass>,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    color_sampler: Option<vk::Sampler>,
    data_sampler: Option<vk::Sampler>,
    uniform_buffers: Vec<GpuBuffer>,
    finished: bool,
}

impl<'a> TemporalBuild<'a> {
    fn new(device: &'a Device) -> Self {
        Self {
            device,
            histories: Vec::new(),
            motion: None,
            render_pass: None,
            framebuffers: Vec::new(),
            pipeline: None,
            pipeline_layout: None,
            descriptor_set_layout: None,
            descriptor_pool: None,
            descriptor_sets: Vec::new(),
            color_sampler: None,
            data_sampler: None,
            uniform_buffers: Vec::new(),
            finished: false,
        }
    }

    fn finish(mut self) -> TemporalAntiAliasing {
        let taa = TemporalAntiAliasing {
            enabled: true,
            histories: std::mem::take(&mut self.histories),
            motion: self
                .motion
                .take()
                .expect("TAA motion target was not created"),
            render_pass: self
                .render_pass
                .take()
                .expect("TAA render pass was not created"),
            framebuffers: std::mem::take(&mut self.framebuffers),
            pipeline: self.pipeline.take().expect("TAA pipeline was not created"),
            pipeline_layout: self
                .pipeline_layout
                .take()
                .expect("TAA pipeline layout was not created"),
            descriptor_set_layout: self
                .descriptor_set_layout
                .take()
                .expect("TAA descriptor set layout was not created"),
            descriptor_pool: self
                .descriptor_pool
                .take()
                .expect("TAA descriptor pool was not created"),
            descriptor_sets: std::mem::take(&mut self.descriptor_sets),
            color_sampler: self
                .color_sampler
                .take()
                .expect("TAA color sampler was not created"),
            data_sampler: self
                .data_sampler
                .take()
                .expect("TAA data sampler was not created"),
            uniform_buffers: std::mem::take(&mut self.uniform_buffers),
            history_write_index: 0,
            history_valid: false,
            stable_metadata_valid: false,
            frame_index: 0,
            previous: None,
            pending: None,
        };
        self.finished = true;
        taa
    }
}

impl Drop for TemporalBuild<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if let Some(pipeline) = self.pipeline.take() {
            destroy_pipeline(self.device, pipeline);
        }
        if let Some(pool) = self.descriptor_pool.take() {
            destroy_descriptor_pool(self.device, pool);
        }
        if let Some(sampler) = self.data_sampler.take() {
            destroy_sampler(self.device, sampler);
        }
        if let Some(sampler) = self.color_sampler.take() {
            destroy_sampler(self.device, sampler);
        }
        if let Some(layout) = self.pipeline_layout.take() {
            destroy_pipeline_layout(self.device, layout);
        }
        if let Some(layout) = self.descriptor_set_layout.take() {
            destroy_descriptor_set_layout(self.device, layout);
        }
        for framebuffer in self.framebuffers.drain(..) {
            destroy_framebuffer(self.device, framebuffer);
        }
        if let Some(render_pass) = self.render_pass.take() {
            destroy_render_pass(self.device, render_pass);
        }
        for buffer in self.uniform_buffers.drain(..) {
            buffer.destroy(self.device);
        }
        if let Some(motion) = self.motion.take() {
            destroy_color_target(self.device, motion.color);
        }
        for history in self.histories.drain(..) {
            destroy_temporal_history(self.device, history);
        }
    }
}

impl TemporalAntiAliasing {
    fn disabled() -> Self {
        Self {
            enabled: false,
            histories: Vec::new(),
            motion: TemporalMotionTarget {
                color: ColorTarget {
                    image: vk::Image::null(),
                    memory: vk::DeviceMemory::null(),
                    view: vk::ImageView::null(),
                    sampled_view: vk::ImageView::null(),
                    mip_views: Vec::new(),
                    format: vk::Format::UNDEFINED,
                },
                state: ResourceState::Undefined,
            },
            render_pass: vk::RenderPass::null(),
            framebuffers: Vec::new(),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
            color_sampler: vk::Sampler::null(),
            data_sampler: vk::Sampler::null(),
            uniform_buffers: Vec::new(),
            history_write_index: 0,
            history_valid: false,
            stable_metadata_valid: false,
            frame_index: 0,
            previous: None,
            pending: None,
        }
    }

    pub(super) fn create(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        extent: NonZeroExtent,
        frame_slot_count: usize,
        enabled: bool,
        current_color_view: vk::ImageView,
        current_depth_view: vk::ImageView,
        current_normal_view: vk::ImageView,
        current_transparent_normal_view: vk::ImageView,
    ) -> Result<Self, VulkanError> {
        if !enabled {
            return Ok(Self::disabled());
        }
        let mut build = TemporalBuild::new(device);
        for _ in 0..TAA_HISTORY_COUNT {
            build
                .histories
                .push(create_temporal_history(device, memory_properties, extent)?);
        }
        build.motion = Some(TemporalMotionTarget {
            color: create_color_target(
                device,
                memory_properties,
                extent,
                TAA_MOTION_FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            )?,
            state: ResourceState::Undefined,
        });
        build.render_pass = Some(create_render_pass(device)?);
        let render_pass = build.render_pass.expect("TAA render pass was just created");
        let motion_view = build
            .motion
            .as_ref()
            .expect("TAA motion target was just created")
            .color
            .view;
        for history in &build.histories {
            build.framebuffers.push(create_framebuffer(
                device,
                render_pass,
                extent,
                history,
                motion_view,
            )?);
        }

        build.descriptor_set_layout = Some(create_descriptor_set_layout(device)?);
        let descriptor_set_layout = build
            .descriptor_set_layout
            .expect("TAA descriptor layout was just created");
        build.pipeline_layout = Some(create_pipeline_layout(device, descriptor_set_layout)?);
        build.color_sampler = Some(create_sampler(device, vk::Filter::LINEAR)?);
        build.data_sampler = Some(create_sampler(device, vk::Filter::NEAREST)?);

        let frame_slot_count = frame_slot_count.max(1);
        for _ in 0..frame_slot_count {
            build.uniform_buffers.push(create_host_buffer(
                device,
                memory_properties,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                size_of::<TemporalResolveUniform>() as vk::DeviceSize,
            )?);
        }
        let descriptor_set_count = CURRENT_COLOR_INPUT_COUNT * TAA_HISTORY_COUNT * frame_slot_count;
        build.descriptor_pool = Some(create_descriptor_pool(device, descriptor_set_count as u32)?);
        build.descriptor_sets = allocate_descriptor_sets(
            device,
            build
                .descriptor_pool
                .expect("TAA descriptor pool was just created"),
            descriptor_set_layout,
            descriptor_set_count,
        )?;
        update_descriptor_sets(
            device,
            &build.descriptor_sets,
            &build.uniform_buffers,
            &build.histories,
            frame_slot_count,
            current_color_view,
            current_depth_view,
            current_normal_view,
            current_transparent_normal_view,
            build
                .color_sampler
                .expect("TAA color sampler was just created"),
            build
                .data_sampler
                .expect("TAA data sampler was just created"),
        );
        build.pipeline = Some(create_pipeline(
            device,
            build
                .pipeline_layout
                .expect("TAA pipeline layout was just created"),
            render_pass,
        )?);

        tracing::info!(
            width = extent.width(),
            height = extent.height(),
            frame_slot_count,
            history_format = ?TAA_HISTORY_FORMAT,
            motion_format = ?TAA_MOTION_FORMAT,
            "created HDR temporal anti-aliasing resources"
        );
        Ok(build.finish())
    }

    pub(super) fn prepare_frame(
        &mut self,
        device: &Device,
        slot_index: usize,
        snapshot: &FrameSnapshot,
        camera: CameraSnapshot,
        quality: RenderQualitySettings,
        extent: vk::Extent2D,
        use_corrected_scene_color: bool,
        volumetric_input: bool,
        temporal_aa_enabled: bool,
    ) -> Result<TaaFrameInfo, VulkanError> {
        let aspect = extent.width.max(1) as f32 / extent.height.max(1) as f32;
        if !temporal_aa_enabled {
            // SMAA consumes the current full-resolution scene directly. Avoid touching the TAA
            // uniform ring or creating a pending history transaction when the temporal resolver
            // is not part of the frame graph; this keeps the dormant compatibility resources out
            // of the normal frame's CPU/GPU work while still returning the camera matrix used by
            // scene and volumetric passes.
            let current_view_projection = camera.view_projection(aspect);
            let inverse_current_view_projection =
                invert_mat4(current_view_projection).unwrap_or_else(identity_mat4);
            let inverse_current_view =
                invert_mat4(camera_view_matrix(camera)).unwrap_or_else(identity_mat4);
            self.pending = None;
            self.stable_metadata_valid = false;
            return Ok(TaaFrameInfo {
                jittered_view_projection: current_view_projection,
                inverse_current_view_projection,
                previous_view_projection: current_view_projection,
                inverse_current_view,
                inverse_previous_view: inverse_current_view,
                current_jitter_pixels: [0.0, 0.0],
                previous_jitter_pixels: [0.0, 0.0],
                write_history_index: self.history_write_index,
                reset_reprojection_history: true,
            });
        }
        let aa_blend = quality.anti_aliasing().blend();
        // Temporal AA is an explicit temporal feature, not a side effect of SMAA's continuous
        // neighborhood blend. `volumetric_input` remains part of the signature for the dormant
        // compatibility route, but it must not turn a spatial quality scalar into an enable flag.
        let temporal_enabled = temporal_aa_enabled;
        let jitter_pixels = if temporal_enabled {
            halton_jitter(self.frame_index % JITTER_SAMPLE_COUNT)
        } else {
            [0.0, 0.0]
        };
        let current_view_projection =
            jitter_view_projection(camera.view_projection(aspect), jitter_pixels, extent);
        let inverse_current_view_projection =
            invert_mat4(current_view_projection).unwrap_or_else(identity_mat4);
        let reset_reprojection_history = !self.history_valid
            || self.previous.is_none_or(|previous| {
                temporal_reprojection_history_discontinuous(previous, snapshot, camera, aspect)
            });
        let volumetric_signature = volumetric_history_signature(quality, volumetric_input);
        let current_light = directional_light_for_snapshot(snapshot);
        let current_light_direction = normalize3(current_light.direction);
        let volumetric_light_change = if temporal_enabled {
            self.previous.map_or(1.0, |previous| {
                volumetric_light_change(
                    current_light_direction,
                    current_light.intensity,
                    current_light.color,
                    previous,
                )
            })
        } else {
            0.0
        };
        let volumetric_history_reset = volumetric_input
            && self.previous.is_none_or(|previous| {
                volumetric_history_discontinuous(
                    previous,
                    snapshot,
                    camera,
                    aspect,
                    volumetric_signature,
                )
            });
        // `volumetric_light_change` is intentionally normalized to a sub-milliradian response
        // range. It is therefore unsuitable as a history-reset predicate: an animated Sun can
        // report one on every frame even though the shadow field is changing continuously. Only a
        // large direction/intensity/color replacement is a discontinuity; ordinary motion remains
        // in the exponential response path below.
        let light_history_reset = temporal_enabled
            && directional_light_history_cut(
                current_light_direction,
                current_light.intensity,
                current_light.color,
                self.previous,
            );
        let reset_history = reset_reprojection_history
            || volumetric_history_reset
            || light_history_reset
            || self.previous.is_none_or(|previous| {
                (aa_blend - previous.aa_blend).abs() > 0.0001
                    || previous.volumetric_input != volumetric_input
                    || (previous.aa_blend > 0.0 || previous.volumetric_input) != temporal_enabled
            });
        self.stable_metadata_valid = self.history_valid && !reset_reprojection_history;
        let previous_view_projection = self
            .previous
            .map_or(current_view_projection, |previous| previous.view_projection);
        let previous_camera = self.previous.map_or(camera, |previous| previous.camera);
        let inverse_current_view =
            invert_mat4(camera_view_matrix(camera)).unwrap_or_else(identity_mat4);
        let inverse_previous_view =
            invert_mat4(camera_view_matrix(previous_camera)).unwrap_or_else(identity_mat4);
        let previous_jitter = self
            .previous
            .map_or(jitter_pixels, |previous| previous.jitter_pixels);
        let feedback = if temporal_enabled {
            exponential_history_feedback(aa_blend, volumetric_input, snapshot.frame_rate_hz)
        } else {
            0.0
        };
        let uniform = TemporalResolveUniform {
            current_view_projection,
            inverse_current_view_projection,
            previous_view_projection,
            inverse_current_view,
            inverse_previous_view,
            texel_feedback_reset: [
                1.0 / extent.width.max(1) as f32,
                1.0 / extent.height.max(1) as f32,
                feedback,
                if reset_history || !temporal_enabled {
                    1.0
                } else {
                    0.0
                },
            ],
            rejection: [
                DEPTH_REJECT_ABSOLUTE,
                DEPTH_REJECT_RELATIVE,
                NORMAL_REJECT_COSINE,
                HISTORY_SAMPLE_LIMIT,
            ],
            jitter_pixels: [
                jitter_pixels[0],
                jitter_pixels[1],
                previous_jitter[0],
                previous_jitter[1],
            ],
            effects: [
                if volumetric_input { 1.0 } else { 0.0 },
                if volumetric_input {
                    VOLUMETRIC_HISTORY_SAMPLE_LIMIT
                } else {
                    0.0
                },
                volumetric_light_change,
                0.0,
            ],
        };
        let uniform_buffer = self.uniform_buffers.get(slot_index).ok_or(
            VulkanError::SwapchainImageIndexOutOfRange {
                index: slot_index,
                count: self.uniform_buffers.len(),
            },
        )?;
        write_buffer_value(device, uniform_buffer, &uniform)?;

        let pending_previous = PreviousFrame {
            frame_id: snapshot.frame_id.raw(),
            scene: snapshot.scene.raw(),
            surface_generation: snapshot.surface_generation.raw(),
            camera,
            aspect,
            view_projection: current_view_projection,
            jitter_pixels,
            aa_blend,
            volumetric_input,
            volumetric_signature,
            light_direction: current_light_direction,
            light_intensity: current_light.intensity,
            light_color: current_light.color,
        };
        self.pending = Some(PendingFrame {
            previous: pending_previous,
            slot_index,
            write_history_index: self.history_write_index,
            current_color_input_index: if use_corrected_scene_color {
                CORRECTED_SCENE_COLOR_INPUT_INDEX
            } else {
                SCENE_COLOR_INPUT_INDEX
            },
        });

        Ok(TaaFrameInfo {
            jittered_view_projection: current_view_projection,
            inverse_current_view_projection,
            previous_view_projection,
            inverse_current_view,
            inverse_previous_view,
            current_jitter_pixels: jitter_pixels,
            previous_jitter_pixels: previous_jitter,
            write_history_index: self.history_write_index,
            reset_reprojection_history,
        })
    }

    pub(super) fn record(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
    ) -> Result<(), VulkanError> {
        if !self.enabled {
            return Err(VulkanError::GraphCompile(
                "TAA pass recorded while temporal AA is disabled".to_string(),
            ));
        }
        let pending = self.pending.ok_or_else(|| {
            VulkanError::GraphCompile("TAA pass recorded before frame preparation".to_string())
        })?;
        let framebuffer = self
            .framebuffers
            .get(pending.write_history_index)
            .copied()
            .ok_or(VulkanError::SwapchainImageIndexOutOfRange {
                index: pending.write_history_index,
                count: self.framebuffers.len(),
            })?;
        let descriptor_index = taa_descriptor_index(
            pending.current_color_input_index,
            pending.write_history_index,
            pending.slot_index,
            self.uniform_buffers.len(),
        );
        let descriptor_set = self.descriptor_sets.get(descriptor_index).copied().ok_or(
            VulkanError::SwapchainImageIndexOutOfRange {
                index: descriptor_index,
                count: self.descriptor_sets.len(),
            },
        )?;
        let render_area = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent);
        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area);
        let viewports = [vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];
        let scissors = [render_area];
        let descriptor_sets = [descriptor_set];

        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            device.cmd_set_viewport(command_buffer, 0, &viewports);
            device.cmd_set_scissor(command_buffer, 0, &scissors);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                shader_interface::FRAME_SET,
                &descriptor_sets,
                &[],
            );
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(command_buffer);
        }
        Ok(())
    }

    /// Binds the corrected-HDR descriptor variants after the shadow resolver creates its target.
    /// The SceneColor variants remain intact for frame graphs that omit scene metadata.
    #[allow(dead_code)]
    pub(super) fn update_current_color_input(
        &self,
        device: &Device,
        current_color_view: vk::ImageView,
    ) {
        if !self.enabled {
            return;
        }
        update_current_color_descriptor_bindings(
            device,
            &self.descriptor_sets,
            self.uniform_buffers.len(),
            current_color_view,
            self.color_sampler,
        );
    }

    #[allow(dead_code)]
    pub(super) fn history_views(&self) -> [vk::ImageView; TAA_HISTORY_COUNT] {
        if !self.enabled {
            return [vk::ImageView::null(); TAA_HISTORY_COUNT];
        }
        std::array::from_fn(|index| self.histories[index].color.view)
    }

    pub(super) fn depth_history_views(&self) -> [vk::ImageView; TAA_HISTORY_COUNT] {
        if !self.enabled {
            return [vk::ImageView::null(); TAA_HISTORY_COUNT];
        }
        std::array::from_fn(|index| self.histories[index].linear_depth.view)
    }

    #[allow(dead_code)]
    pub(super) fn normal_history_views(&self) -> [vk::ImageView; TAA_HISTORY_COUNT] {
        if !self.enabled {
            return [vk::ImageView::null(); TAA_HISTORY_COUNT];
        }
        std::array::from_fn(|index| self.histories[index].normal.view)
    }

    pub(super) fn history_write_index(&self) -> usize {
        self.history_write_index
    }

    pub(super) fn stable_metadata_valid(&self) -> bool {
        self.stable_metadata_valid
    }

    /// Returns the jitter used by the frame currently being recorded. Post effects reconstructing
    /// positions from the jittered scene depth must use this exact sample offset.
    pub(super) fn pending_jitter_pixels(&self) -> [f32; 2] {
        self.pending
            .map(|pending| pending.previous.jitter_pixels)
            .unwrap_or([0.0, 0.0])
    }

    pub(super) fn graph_states(
        &self,
    ) -> (
        [ResourceState; TAA_HISTORY_COUNT],
        [ResourceState; TAA_HISTORY_COUNT],
        [ResourceState; TAA_HISTORY_COUNT],
        ResourceState,
    ) {
        if !self.enabled {
            return (
                [ResourceState::Undefined; TAA_HISTORY_COUNT],
                [ResourceState::Undefined; TAA_HISTORY_COUNT],
                [ResourceState::Undefined; TAA_HISTORY_COUNT],
                ResourceState::Undefined,
            );
        }
        (
            std::array::from_fn(|index| self.histories[index].color_state),
            std::array::from_fn(|index| self.histories[index].depth_state),
            std::array::from_fn(|index| self.histories[index].normal_state),
            self.motion.state,
        )
    }

    pub(super) fn apply_graph_final_states(&mut self, plan: &FrameGraphPlan) {
        if !self.enabled {
            self.pending = None;
            self.history_valid = false;
            self.stable_metadata_valid = false;
            return;
        }
        for (index, history) in self.histories.iter_mut().enumerate() {
            if let Some(state) = plan.final_state_for(TAA_HISTORY_RESOURCES[index]) {
                history.color_state = state;
            }
            if let Some(state) = plan.final_state_for(TAA_DEPTH_HISTORY_RESOURCES[index]) {
                history.depth_state = state;
            }
            if let Some(state) = plan.final_state_for(TAA_NORMAL_HISTORY_RESOURCES[index]) {
                history.normal_state = state;
            }
        }
        if let Some(state) = plan.final_state_for(TAA_MOTION_RESOURCE) {
            self.motion.state = state;
        }
        if plan
            .passes()
            .iter()
            .any(|pass| pass.name() == TAA_RESOLVE_PASS)
        {
            if let Some(pending) = self.pending.take() {
                self.previous = Some(pending.previous);
                self.history_valid = true;
                self.stable_metadata_valid = true;
                self.history_write_index = 1 - pending.write_history_index;
                self.frame_index = self.frame_index.saturating_add(1);
            }
        } else {
            self.pending = None;
            self.history_valid = false;
            self.stable_metadata_valid = false;
        }
    }

    pub(super) fn graph_image(
        &self,
        resource: GraphResource,
    ) -> Option<(vk::Image, vk::ImageAspectFlags)> {
        if !self.enabled {
            return None;
        }
        let image = if let Some(index) = resource.taa_history() {
            self.histories.get(index).map(|target| target.color.image)
        } else if let Some(index) = resource.taa_depth_history() {
            self.histories
                .get(index)
                .map(|target| target.linear_depth.image)
        } else if let Some(index) = resource.taa_normal_history() {
            self.histories.get(index).map(|target| target.normal.image)
        } else if resource == TAA_MOTION_RESOURCE {
            Some(self.motion.color.image)
        } else {
            None
        }?;
        Some((image, vk::ImageAspectFlags::COLOR))
    }

    pub(super) fn destroy(self, device: &Device) {
        if !self.enabled {
            return;
        }
        destroy_pipeline(device, self.pipeline);
        destroy_descriptor_pool(device, self.descriptor_pool);
        destroy_sampler(device, self.data_sampler);
        destroy_sampler(device, self.color_sampler);
        destroy_pipeline_layout(device, self.pipeline_layout);
        destroy_descriptor_set_layout(device, self.descriptor_set_layout);
        for framebuffer in self.framebuffers {
            destroy_framebuffer(device, framebuffer);
        }
        destroy_render_pass(device, self.render_pass);
        for buffer in self.uniform_buffers {
            buffer.destroy(device);
        }
        destroy_color_target(device, self.motion.color);
        for history in self.histories {
            destroy_temporal_history(device, history);
        }
    }
}

fn create_temporal_history(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    extent: NonZeroExtent,
) -> Result<TemporalHistoryTarget, VulkanError> {
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    let color = create_color_target(device, memory_properties, extent, TAA_HISTORY_FORMAT, usage)?;
    let linear_depth = match create_color_target(
        device,
        memory_properties,
        extent,
        TAA_LINEAR_DEPTH_FORMAT,
        usage,
    ) {
        Ok(target) => target,
        Err(error) => {
            destroy_color_target(device, color);
            return Err(error);
        }
    };
    let normal =
        match create_color_target(device, memory_properties, extent, TAA_NORMAL_FORMAT, usage) {
            Ok(target) => target,
            Err(error) => {
                destroy_color_target(device, linear_depth);
                destroy_color_target(device, color);
                return Err(error);
            }
        };
    Ok(TemporalHistoryTarget {
        color,
        linear_depth,
        normal,
        color_state: ResourceState::Undefined,
        depth_state: ResourceState::Undefined,
        normal_state: ResourceState::Undefined,
    })
}

fn destroy_temporal_history(device: &Device, history: TemporalHistoryTarget) {
    destroy_color_target(device, history.normal);
    destroy_color_target(device, history.linear_depth);
    destroy_color_target(device, history.color);
}

fn create_render_pass(device: &Device) -> Result<vk::RenderPass, VulkanError> {
    let formats = [
        TAA_HISTORY_FORMAT,
        TAA_MOTION_FORMAT,
        TAA_LINEAR_DEPTH_FORMAT,
        TAA_NORMAL_FORMAT,
    ];
    let attachments = formats.map(|format| {
        vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    });
    let color_references = std::array::from_fn::<_, 4, _>(|index| {
        vk::AttachmentReference::default()
            .attachment(index as u32)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    });
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_references)];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    unsafe { device.create_render_pass(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    extent: NonZeroExtent,
    history: &TemporalHistoryTarget,
    motion_view: vk::ImageView,
) -> Result<vk::Framebuffer, VulkanError> {
    let attachments = [
        history.color.view,
        motion_view,
        history.linear_depth.view,
        history.normal.view,
    ];
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(extent.width())
        .height(extent.height())
        .layers(1);
    unsafe { device.create_framebuffer(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let sampler_binding = |binding| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
    };
    let bindings = [
        sampler_binding(CURRENT_COLOR_BINDING),
        sampler_binding(CURRENT_DEPTH_BINDING),
        sampler_binding(CURRENT_NORMAL_BINDING),
        sampler_binding(PREVIOUS_COLOR_BINDING),
        sampler_binding(PREVIOUS_DEPTH_BINDING),
        sampler_binding(PREVIOUS_NORMAL_BINDING),
        sampler_binding(CURRENT_TRANSPARENT_NORMAL_BINDING),
        vk::DescriptorSetLayoutBinding::default()
            .binding(PARAMS_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, VulkanError> {
    let set_layouts = [descriptor_set_layout];
    let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    unsafe { device.create_pipeline_layout(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_sampler(device: &Device, filter: vk::Filter) -> Result<vk::Sampler, VulkanError> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(0.0);
    unsafe { device.create_sampler(&create_info, None) }.map_err(VulkanError::Vk)
}

fn create_descriptor_pool(
    device: &Device,
    set_count: u32,
) -> Result<vk::DescriptorPool, VulkanError> {
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(set_count * 7),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(set_count),
    ];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(set_count)
        .pool_sizes(&pool_sizes);
    unsafe { device.create_descriptor_pool(&create_info, None) }.map_err(VulkanError::Vk)
}

fn allocate_descriptor_sets(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    count: usize,
) -> Result<Vec<vk::DescriptorSet>, VulkanError> {
    let layouts = vec![layout; count];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    unsafe { device.allocate_descriptor_sets(&allocate_info) }.map_err(VulkanError::Vk)
}

#[allow(clippy::too_many_arguments)]
fn update_descriptor_sets(
    device: &Device,
    descriptor_sets: &[vk::DescriptorSet],
    uniform_buffers: &[GpuBuffer],
    histories: &[TemporalHistoryTarget],
    frame_slot_count: usize,
    current_color_view: vk::ImageView,
    current_depth_view: vk::ImageView,
    current_normal_view: vk::ImageView,
    current_transparent_normal_view: vk::ImageView,
    color_sampler: vk::Sampler,
    data_sampler: vk::Sampler,
) {
    for current_color_input_index in 0..CURRENT_COLOR_INPUT_COUNT {
        for write_index in 0..TAA_HISTORY_COUNT {
            let read_index = 1 - write_index;
            for (slot_index, uniform_buffer) in uniform_buffers.iter().enumerate() {
                let descriptor_index = taa_descriptor_index(
                    current_color_input_index,
                    write_index,
                    slot_index,
                    frame_slot_count,
                );
                let descriptor_set = descriptor_sets[descriptor_index];
                let current_color = [image_info(color_sampler, current_color_view)];
                let current_depth = [image_info(data_sampler, current_depth_view)];
                let current_normal = [image_info(data_sampler, current_normal_view)];
                let previous_color = [image_info(color_sampler, histories[read_index].color.view)];
                let previous_depth = [image_info(
                    data_sampler,
                    histories[read_index].linear_depth.view,
                )];
                let previous_normal = [image_info(data_sampler, histories[read_index].normal.view)];
                let current_transparent_normal =
                    [image_info(data_sampler, current_transparent_normal_view)];
                let buffer_info = [vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffer.handle())
                    .offset(0)
                    .range(size_of::<TemporalResolveUniform>() as vk::DeviceSize)];
                let writes = [
                    image_write(descriptor_set, CURRENT_COLOR_BINDING, &current_color),
                    image_write(descriptor_set, CURRENT_DEPTH_BINDING, &current_depth),
                    image_write(descriptor_set, CURRENT_NORMAL_BINDING, &current_normal),
                    image_write(descriptor_set, PREVIOUS_COLOR_BINDING, &previous_color),
                    image_write(descriptor_set, PREVIOUS_DEPTH_BINDING, &previous_depth),
                    image_write(descriptor_set, PREVIOUS_NORMAL_BINDING, &previous_normal),
                    image_write(
                        descriptor_set,
                        CURRENT_TRANSPARENT_NORMAL_BINDING,
                        &current_transparent_normal,
                    ),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(PARAMS_BINDING)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buffer_info),
                ];
                unsafe {
                    device.update_descriptor_sets(&writes, &[]);
                }
            }
        }
    }
}

#[allow(dead_code)]
fn update_current_color_descriptor_bindings(
    device: &Device,
    descriptor_sets: &[vk::DescriptorSet],
    frame_slot_count: usize,
    current_color_view: vk::ImageView,
    color_sampler: vk::Sampler,
) {
    for taa_write_index in 0..TAA_HISTORY_COUNT {
        for slot_index in 0..frame_slot_count {
            let descriptor_set = descriptor_sets[taa_descriptor_index(
                CORRECTED_SCENE_COLOR_INPUT_INDEX,
                taa_write_index,
                slot_index,
                frame_slot_count,
            )];
            let current_color = [image_info(color_sampler, current_color_view)];
            let writes = [image_write(
                descriptor_set,
                CURRENT_COLOR_BINDING,
                &current_color,
            )];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }
}

fn taa_descriptor_index(
    current_color_input_index: usize,
    taa_write_index: usize,
    slot_index: usize,
    frame_slot_count: usize,
) -> usize {
    ((current_color_input_index * TAA_HISTORY_COUNT + taa_write_index) * frame_slot_count)
        + slot_index
}

fn image_info(sampler: vk::Sampler, view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
}

fn image_write<'a>(
    descriptor_set: vk::DescriptorSet,
    binding: u32,
    image_info: &'a [vk::DescriptorImageInfo],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(image_info)
}

fn create_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> Result<vk::Pipeline, VulkanError> {
    let vertex_shader = shader::create_shader_module(device, VERTEX_SHADER)?;
    let fragment_shader = match shader::create_shader_module(device, FRAGMENT_SHADER) {
        Ok(module) => module,
        Err(error) => {
            shader::destroy_shader_module(device, vertex_shader);
            return Err(error);
        }
    };
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(SHADER_ENTRY),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(SHADER_ENTRY),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_write = vk::PipelineColorBlendAttachmentState::default().color_write_mask(
        vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A,
    );
    let color_attachments = [color_write; 4];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline_infos = [pipeline_info];
    let result = match unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
    } {
        Ok(mut pipelines) => Ok(pipelines.remove(0)),
        Err((pipelines, error)) => {
            for pipeline in pipelines {
                destroy_pipeline(device, pipeline);
            }
            Err(VulkanError::Vk(error))
        }
    };
    shader::destroy_shader_module(device, fragment_shader);
    shader::destroy_shader_module(device, vertex_shader);
    result
}

fn temporal_reprojection_history_discontinuous(
    previous: PreviousFrame,
    snapshot: &FrameSnapshot,
    camera: CameraSnapshot,
    aspect: f32,
) -> bool {
    // A skipped application frame is still valid: `previous` is the last frame actually rendered,
    // and its exact matrices remain the right reprojection source. Only stale/non-monotonic input
    // invalidates that relationship.
    if snapshot.frame_id.raw() <= previous.frame_id
        || snapshot.scene.raw() != previous.scene
        || snapshot.surface_generation.raw() != previous.surface_generation
        || (aspect - previous.aspect).abs() > 0.0005
    {
        return true;
    }

    let eye_delta = distance3(camera.eye, previous.camera.eye);
    let view_distance = distance3(previous.camera.eye, previous.camera.target).max(1.0);
    let forward = normalize3(sub3(camera.target, camera.eye));
    let previous_forward = normalize3(sub3(previous.camera.target, previous.camera.eye));
    let direction_cosine = dot3(forward, previous_forward);
    let fov_delta = (camera.fov_y_radians - previous.camera.fov_y_radians).abs();
    let near_delta = (camera.near - previous.camera.near).abs() / previous.camera.near.max(0.0001);
    let far_delta = (camera.far - previous.camera.far).abs() / previous.camera.far.max(0.0001);

    eye_delta > (view_distance * 2.0).max(5.0)
        || direction_cosine < 35.0_f32.to_radians().cos()
        || fov_delta > 8.0_f32.to_radians()
        || near_delta > 0.10
        || far_delta > 0.10
}

/// Returns whether a camera-ray volume can safely reuse the previous full-resolution sample.
///
/// Unlike opaque TAA, a volumetric value has no single world-space anchor. The resolve nevertheless
/// uses the current scene-depth endpoint (or the finite far-plane endpoint for the background) as
/// a continuous ray anchor, so small camera motion can be reprojected instead of resetting the
/// accumulation every frame.  Camera/light/medium discontinuities still start a clean sequence.
fn volumetric_history_discontinuous(
    previous: PreviousFrame,
    snapshot: &FrameSnapshot,
    camera: CameraSnapshot,
    aspect: f32,
    volumetric_signature: u64,
) -> bool {
    if !previous.volumetric_input
        || previous.volumetric_signature != volumetric_signature
        || temporal_reprojection_history_discontinuous(previous, snapshot, camera, aspect)
    {
        return true;
    }

    // `temporal_reprojection_history_discontinuous` already rejects skipped frames, scene/surface
    // changes, aspect changes, large translations (five world units), large rotations (35°),
    // and clip-plane edits.  Smaller motion is handled by the endpoint reprojection in the
    // shader; resetting at a sub-pixel threshold was the source of the severe moving-camera
    // flicker.
    false
}

/// Hashes the medium and quality inputs that define the volumetric lighting contract.
///
/// Directional-light motion is deliberately not part of this reset key.  A moving Sun changes the
/// shadow field continuously, so treating each quantized light step as a history discontinuity
/// would make the TAA pass look disabled during an otherwise smooth day/night animation.  The
/// light is tracked separately by `volumetric_light_change` and only raises the current sample
/// contribution when the change is genuinely large.
fn volumetric_history_signature(quality: RenderQualitySettings, volumetric_input: bool) -> u64 {
    let fog = quality.fog();
    let bloom = quality.bloom();
    let shadow = quality.stable_csm_pcss();
    let values = [
        if volumetric_input { 1.0 } else { 0.0 },
        if fog.enabled() { 1.0 } else { 0.0 },
        fog.density(),
        fog.height_falloff(),
        fog.height(),
        fog.max_distance(),
        bloom.god_rays_intensity(),
        if quality.features().volumetric_fog_enabled() {
            1.0
        } else {
            0.0
        },
        shadow.light_angular_radius_radians(),
        shadow.receiver_bias_scale(),
        shadow.slope_bias_scale(),
        shadow.normal_offset_scale(),
        shadow.receiver_plane_bias_scale(),
        shadow.blocker_search_samples() as f32,
        shadow.filter_samples() as f32,
        shadow.shadow_map_resolution() as f32,
    ];
    let mut key = 0xcbf29ce484222325_u64;
    for value in values {
        key ^= u64::from(value.to_bits());
        key = key.wrapping_mul(0x100000001b3);
    }
    key
}

/// Returns whether the directional light was replaced rather than merely moved.
///
/// The renderer also computes `volumetric_light_change`, but that value is normalized to a very
/// small angular step so a moving CSM field can receive a little more current-frame weight. Using
/// that sensitive signal to clear TAA would make an animated Sun clear the history on every frame
/// and expose raw shadow-map quantization as pixel flicker. Keep reset thresholds intentionally
/// coarse and reserve them for a real light cut (day/night swap, intensity edit, or color edit).
fn directional_light_history_cut(
    current_direction: [f32; 3],
    current_intensity: f32,
    current_color: [f32; 3],
    previous: Option<PreviousFrame>,
) -> bool {
    let Some(previous) = previous else {
        return false;
    };

    let previous_direction = normalize3(previous.light_direction);
    let direction_cosine = dot3(current_direction, previous_direction).clamp(-1.0, 1.0);
    let direction_cross = cross3(current_direction, previous_direction);
    let direction_sine = dot3(direction_cross, direction_cross)
        .sqrt()
        .clamp(0.0, 1.0);
    let angular_change = direction_sine.atan2(direction_cosine).abs();
    if angular_change > TAA_LIGHT_CUT_ANGLE_RADIANS {
        return true;
    }

    let current_intensity = if current_intensity.is_finite() {
        current_intensity.max(0.0)
    } else {
        0.0
    };
    let previous_intensity = if previous.light_intensity.is_finite() {
        previous.light_intensity.max(0.0)
    } else {
        0.0
    };
    let intensity_max = current_intensity.max(previous_intensity);
    let intensity_min = current_intensity.min(previous_intensity);
    if intensity_max > 0.25
        && intensity_max / intensity_min.max(0.01) > TAA_LIGHT_CUT_INTENSITY_RATIO
    {
        return true;
    }

    let color_delta = current_color
        .into_iter()
        .zip(previous.light_color)
        .map(|(current, previous)| {
            let current = if current.is_finite() {
                current.max(0.0)
            } else {
                0.0
            };
            let previous = if previous.is_finite() {
                previous.max(0.0)
            } else {
                0.0
            };
            (current - previous).abs() / current.max(previous).max(0.05)
        })
        .fold(0.0_f32, f32::max);
    color_delta > TAA_LIGHT_CUT_COLOR_DELTA
}

/// Returns the directional Sun used by the frame, matching the renderer's global-light fallback.
fn directional_light_for_snapshot(snapshot: &FrameSnapshot) -> LightPacket {
    snapshot
        .lights
        .first()
        .copied()
        .unwrap_or_else(|| LightPacket::new(1.0))
}

/// Converts a raw directional-light change into a bounded temporal response signal.
///
/// The volumetric shadow field has no per-pixel velocity: changing the Sun moves a shadow silhouette
/// even when the endpoint reprojection is exactly stationary.  Quantized history signatures only
/// detect larger epochs, so this continuous signal is used by the shader to raise the current
/// sample weight between those epochs.  Direction, intensity, and chromatic changes share one
/// maximum because each can invalidate the old scattering source.
fn volumetric_light_change(
    current_direction: [f32; 3],
    current_intensity: f32,
    current_color: [f32; 3],
    previous: PreviousFrame,
) -> f32 {
    light_motion_signal(
        current_direction,
        current_intensity,
        current_color,
        previous.light_direction,
        previous.light_intensity,
        previous.light_color,
    )
}

/// Converts the directional-light change tracked by the dedicated volumetric/PCSS state into the
/// same bounded response used by the dormant full-resolution TAA route.
fn temporal_effect_light_change(light: LightPacket, previous: TemporalEffectPrevious) -> f32 {
    light_motion_signal(
        normalize3(light.direction),
        light.intensity,
        light.color,
        previous.light_direction,
        previous.light_intensity,
        previous.light_color,
    )
}

/// Converts camera translation/rotation between two rendered frames into a bounded PCSS history
/// reactivity signal. This is deliberately continuous: the existing discontinuity predicate still
/// handles cuts, while ordinary camera motion only shortens visibility reuse near moving edges.
fn pcss_camera_motion_signal(current: CameraSnapshot, previous: CameraSnapshot) -> f32 {
    let current_forward = normalize3(sub3(current.target, current.eye));
    let previous_forward = normalize3(sub3(previous.target, previous.eye));
    let direction_cosine = dot3(current_forward, previous_forward).clamp(-1.0, 1.0);
    let direction_cross = cross3(current_forward, previous_forward);
    let direction_sine = dot3(direction_cross, direction_cross)
        .sqrt()
        .clamp(0.0, 1.0);
    let angular_change = direction_sine.atan2(direction_cosine).abs();
    let angular_signal = angular_change / PCSS_CAMERA_ANGLE_FULL_RESPONSE_RADIANS;

    let view_distance = distance3(previous.eye, previous.target).max(1.0);
    let translation_signal = distance3(current.eye, previous.eye)
        / (view_distance * PCSS_CAMERA_TRANSLATION_FULL_RESPONSE_FRACTION).max(0.001);

    angular_signal.max(translation_signal).clamp(0.0, 1.0)
}

fn light_motion_signal(
    current_direction: [f32; 3],
    current_intensity: f32,
    current_color: [f32; 3],
    previous_direction: [f32; 3],
    previous_intensity: f32,
    previous_color: [f32; 3],
) -> f32 {
    let previous_direction = normalize3(previous_direction);
    let direction_cosine = dot3(current_direction, previous_direction).clamp(-1.0, 1.0);
    // `acos(dot(a, b))` loses all sub-milliradian motion in f32 because the dot product rounds to
    // exactly one long before the angle reaches the response threshold.  The cross-product sine
    // retains that small signal; atan2(sin, cos) is also well behaved for a 180-degree light cut.
    let direction_cross = cross3(current_direction, previous_direction);
    let direction_sine = dot3(direction_cross, direction_cross)
        .sqrt()
        .clamp(0.0, 1.0);
    let angular_change = direction_sine.atan2(direction_cosine);
    let direction_signal = angular_change / VOLUMETRIC_LIGHT_ANGLE_FULL_RESPONSE;

    let current_intensity = if current_intensity.is_finite() {
        current_intensity.max(0.0)
    } else {
        0.0
    };
    let previous_intensity = if previous_intensity.is_finite() {
        previous_intensity.max(0.0)
    } else {
        0.0
    };
    let intensity_signal = (current_intensity - previous_intensity).abs()
        / (current_intensity.max(previous_intensity).max(0.25)
            * VOLUMETRIC_LIGHT_INTENSITY_FULL_RESPONSE);

    let color_signal = current_color
        .into_iter()
        .zip(previous_color)
        .map(|(current, previous)| {
            let current = if current.is_finite() {
                current.max(0.0)
            } else {
                0.0
            };
            let previous = if previous.is_finite() {
                previous.max(0.0)
            } else {
                0.0
            };
            (current - previous).abs() / current.max(previous).max(0.05)
        })
        .fold(0.0_f32, f32::max)
        / VOLUMETRIC_LIGHT_COLOR_FULL_RESPONSE;

    direction_signal
        .max(intensity_signal)
        .max(color_signal)
        .clamp(0.0, 1.0)
}

/// Converts the quality profile's reference feedback into a frame-rate-aware exponential rate.
///
/// The shader applies this value recursively, so the history contribution after `k` frames is
/// `feedback^k`.  Calibrating the decay in seconds keeps the visual response stable when the
/// configured presentation rate changes, while `TAA_CONVERGENCE_TIME_SCALE` gives the current
/// frame a deliberately longer convergence tail than the previous count-based profile.
fn exponential_history_feedback(aa_blend: f32, volumetric_input: bool, frame_rate_hz: f32) -> f32 {
    let blend = aa_blend.clamp(0.0, 1.0);
    let reference_feedback = if volumetric_input {
        0.94 + 0.04 * blend
    } else {
        0.80 + 0.16 * blend
    };
    let safe_frame_rate = if frame_rate_hz.is_finite() {
        frame_rate_hz.clamp(15.0, 240.0)
    } else {
        TAA_REFERENCE_FRAME_RATE_HZ
    };
    let reference_decay_per_second = -reference_feedback.ln() * TAA_REFERENCE_FRAME_RATE_HZ;
    let profile_decay_per_second = reference_decay_per_second / TAA_CONVERGENCE_TIME_SCALE.max(1.0);
    let profile_half_life_seconds = TAA_LN_TWO / profile_decay_per_second.max(1.0e-6);
    let half_life_seconds = profile_half_life_seconds.max(TAA_MIN_HALF_LIFE_SECONDS);
    // The configured FPS only changes the size of each discrete step.  The half-life itself stays
    // in seconds, so 60/120/240 Hz retain the same amount of history after one real-time second.
    let decay_per_second = TAA_LN_TWO / half_life_seconds;
    (-decay_per_second / safe_frame_rate)
        .exp()
        .clamp(0.0, 0.995)
}

fn halton_jitter(index: u64) -> [f32; 2] {
    // Start at sample one: Halton sample zero is a corner. Center the complete 16-phase cycle so
    // a converged history has no persistent sub-pixel bias in either axis.
    [
        halton(index + 1, 2) - 0.5 - JITTER_SEQUENCE_CENTROID[0],
        halton(index + 1, 3) - 0.5 - JITTER_SEQUENCE_CENTROID[1],
    ]
}

fn halton(mut index: u64, base: u64) -> f32 {
    let mut fraction = 1.0_f32;
    let mut result = 0.0_f32;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

fn jitter_view_projection(
    mut view_projection: [f32; 16],
    jitter_pixels: [f32; 2],
    extent: vk::Extent2D,
) -> [f32; 16] {
    let jitter_ndc = [
        jitter_pixels[0] * 2.0 / extent.width.max(1) as f32,
        jitter_pixels[1] * 2.0 / extent.height.max(1) as f32,
    ];
    for column in 0..4 {
        let base = column * 4;
        let homogeneous_row = view_projection[base + 3];
        view_projection[base] += jitter_ndc[0] * homogeneous_row;
        view_projection[base + 1] += jitter_ndc[1] * homogeneous_row;
    }
    view_projection
}

fn invert_mat4(matrix: [f32; 16]) -> Option<[f32; 16]> {
    let mut augmented = [[0.0_f32; 8]; 4];
    for row in 0..4 {
        for column in 0..4 {
            augmented[row][column] = matrix[column * 4 + row];
        }
        augmented[row][row + 4] = 1.0;
    }
    for pivot_column in 0..4 {
        let mut pivot_row = pivot_column;
        for row in pivot_column + 1..4 {
            if augmented[row][pivot_column].abs() > augmented[pivot_row][pivot_column].abs() {
                pivot_row = row;
            }
        }
        if augmented[pivot_row][pivot_column].abs() <= 1e-8 {
            return None;
        }
        if pivot_row != pivot_column {
            augmented.swap(pivot_row, pivot_column);
        }
        let inverse_pivot = 1.0 / augmented[pivot_column][pivot_column];
        for column in 0..8 {
            augmented[pivot_column][column] *= inverse_pivot;
        }
        for row in 0..4 {
            if row == pivot_column {
                continue;
            }
            let factor = augmented[row][pivot_column];
            for column in 0..8 {
                augmented[row][column] -= factor * augmented[pivot_column][column];
            }
        }
    }
    let mut inverse = [0.0_f32; 16];
    for row in 0..4 {
        for column in 0..4 {
            inverse[column * 4 + row] = augmented[row][column + 4];
        }
    }
    Some(inverse)
}

fn identity_mat4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn camera_view_matrix(camera: CameraSnapshot) -> [f32; 16] {
    let forward = normalize3(sub3(camera.target, camera.eye));
    let right = normalize_or_axis(cross3(forward, camera.up), [1.0, 0.0, 0.0]);
    let up = cross3(right, forward);
    [
        right[0],
        up[0],
        -forward[0],
        0.0,
        right[1],
        up[1],
        -forward[1],
        0.0,
        right[2],
        up[2],
        -forward[2],
        0.0,
        -dot3(right, camera.eye),
        -dot3(up, camera.eye),
        dot3(forward, camera.eye),
        1.0,
    ]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize_or_axis(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length = dot3(value, value).sqrt();
    if length <= 1e-6 {
        fallback
    } else {
        [value[0] / length, value[1] / length, value[2] / length]
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let length = dot3(value, value).sqrt();
    if length <= 1e-6 {
        [0.0, 0.0, -1.0]
    } else {
        [value[0] / length, value[1] / length, value[2] / length]
    }
}

fn distance3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let delta = sub3(a, b);
    dot3(delta, delta).sqrt()
}

fn destroy_framebuffer(device: &Device, framebuffer: vk::Framebuffer) {
    if framebuffer != vk::Framebuffer::null() {
        unsafe { device.destroy_framebuffer(framebuffer, None) };
    }
}

fn destroy_render_pass(device: &Device, render_pass: vk::RenderPass) {
    if render_pass != vk::RenderPass::null() {
        unsafe { device.destroy_render_pass(render_pass, None) };
    }
}

fn destroy_pipeline(device: &Device, pipeline: vk::Pipeline) {
    if pipeline != vk::Pipeline::null() {
        unsafe { device.destroy_pipeline(pipeline, None) };
    }
}

fn destroy_pipeline_layout(device: &Device, layout: vk::PipelineLayout) {
    if layout != vk::PipelineLayout::null() {
        unsafe { device.destroy_pipeline_layout(layout, None) };
    }
}

fn destroy_descriptor_set_layout(device: &Device, layout: vk::DescriptorSetLayout) {
    if layout != vk::DescriptorSetLayout::null() {
        unsafe { device.destroy_descriptor_set_layout(layout, None) };
    }
}

fn destroy_descriptor_pool(device: &Device, pool: vk::DescriptorPool) {
    if pool != vk::DescriptorPool::null() {
        unsafe { device.destroy_descriptor_pool(pool, None) };
    }
}

fn destroy_sampler(device: &Device, sampler: vk::Sampler) {
    if sampler != vk::Sampler::null() {
        unsafe { device.destroy_sampler(sampler, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        FrameId, FrameSnapshotBuilder, SceneHandle, SurfaceGeneration, SurfaceId, ViewId,
        ViewPacket,
    };

    fn transform_homogeneous(matrix: [f32; 16], value: [f32; 4]) -> [f32; 4] {
        std::array::from_fn(|row| {
            matrix[row] * value[0]
                + matrix[4 + row] * value[1]
                + matrix[8 + row] * value[2]
                + matrix[12 + row] * value[3]
        })
    }

    fn projected_uv_and_depth(matrix: [f32; 16], world: [f32; 3]) -> ([f32; 2], f32) {
        let clip = transform_homogeneous(matrix, [world[0], world[1], world[2], 1.0]);
        assert!(
            clip[3] > 0.0,
            "test point must remain in front of the camera"
        );
        let inverse_w = 1.0 / clip[3];
        (
            [
                clip[0] * inverse_w * 0.5 + 0.5,
                clip[1] * inverse_w * 0.5 + 0.5,
            ],
            clip[2] * inverse_w,
        )
    }

    fn snapshot_with_frame_id(frame_id: u64, camera: CameraSnapshot) -> FrameSnapshot {
        let frame_id = FrameId::from_raw(frame_id).expect("test frame id is non-zero");
        let scene = SceneHandle::from_raw(1).expect("test scene handle is non-zero");
        let surface = SurfaceId::from_raw(1).expect("test surface id is non-zero");
        let generation =
            SurfaceGeneration::from_raw(1).expect("test surface generation is non-zero");
        let view = ViewId::from_raw(1).expect("test view id is non-zero");
        let extent = NonZeroExtent::new(1600, 900).expect("test extent is non-zero");
        let mut builder = FrameSnapshotBuilder::new(frame_id, scene, surface, generation);
        builder.add_view(ViewPacket::new(view, extent).with_camera(camera));
        builder.build().expect("test snapshot has one view")
    }

    #[test]
    fn halton_jitter_stays_inside_one_pixel() {
        let mut centroid = [0.0_f32; 2];
        for index in 0..JITTER_SAMPLE_COUNT {
            let jitter = halton_jitter(index);
            assert!((-0.5..=0.5).contains(&jitter[0]));
            assert!((-0.5..=0.5).contains(&jitter[1]));
            centroid[0] += jitter[0];
            centroid[1] += jitter[1];
        }
        assert!(centroid[0].abs() < 1.0e-6);
        assert!(centroid[1].abs() < 1.0e-6);
    }

    #[test]
    fn jitter_changes_only_clip_xy_rows() {
        let original = identity_mat4();
        let jittered = jitter_view_projection(
            original,
            [0.25, -0.25],
            vk::Extent2D {
                width: 100,
                height: 50,
            },
        );
        assert_eq!(jittered[3], original[3]);
        assert_eq!(jittered[7], original[7]);
        assert_eq!(jittered[11], original[11]);
        assert_eq!(jittered[15], original[15]);
        assert_ne!(jittered[12], original[12]);
        assert_ne!(jittered[13], original[13]);
    }

    #[test]
    fn static_reprojection_keeps_same_world_jitter_delta_and_stable_fallback() {
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let aspect = extent.width as f32 / extent.height as f32;
        let camera = CameraSnapshot::perspective(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            60.0_f32.to_radians(),
            0.1,
            100.0,
        )
        .expect("test camera is valid");
        let current_jitter = halton_jitter(5);
        let previous_jitter = halton_jitter(11);
        let current_view_projection =
            jitter_view_projection(camera.view_projection(aspect), current_jitter, extent);
        let previous_view_projection =
            jitter_view_projection(camera.view_projection(aspect), previous_jitter, extent);
        let inverse_current =
            invert_mat4(current_view_projection).expect("test projection is invertible");
        let world = [0.35, -0.2, 0.0];
        let (current_uv, current_depth) = projected_uv_and_depth(current_view_projection, world);

        // Mirror temporal_reproject_world: reconstruct from the current jittered device sample,
        // then project through the exact previous jittered matrix.
        let current_clip = [
            current_uv[0] * 2.0 - 1.0,
            current_uv[1] * 2.0 - 1.0,
            current_depth,
            1.0,
        ];
        let reconstructed_h = transform_homogeneous(inverse_current, current_clip);
        let inverse_reconstructed_w = 1.0 / reconstructed_h[3];
        let reconstructed_world = [
            reconstructed_h[0] * inverse_reconstructed_w,
            reconstructed_h[1] * inverse_reconstructed_w,
            reconstructed_h[2] * inverse_reconstructed_w,
        ];
        let (matrix_previous_uv, _) =
            projected_uv_and_depth(previous_view_projection, reconstructed_world);
        let texel_size = [1.0 / extent.width as f32, 1.0 / extent.height as f32];
        let same_world_previous_uv = matrix_previous_uv;
        let stable_previous_uv = [
            matrix_previous_uv[0] + (current_jitter[0] - previous_jitter[0]) * texel_size[0],
            matrix_previous_uv[1] + (current_jitter[1] - previous_jitter[1]) * texel_size[1],
        ];

        assert!(
            (matrix_previous_uv[0] - current_uv[0]).abs() > 1.0e-5
                || (matrix_previous_uv[1] - current_uv[1]).abs() > 1.0e-5,
            "different jitter phases must produce a measurable raw reprojection delta"
        );
        assert!(
            (same_world_previous_uv[0]
                - current_uv[0]
                - (previous_jitter[0] - current_jitter[0]) * texel_size[0])
                .abs()
                < 1.0e-5
        );
        assert!(
            (same_world_previous_uv[1]
                - current_uv[1]
                - (previous_jitter[1] - current_jitter[1]) * texel_size[1])
                .abs()
                < 1.0e-5
        );
        assert!((stable_previous_uv[0] - current_uv[0]).abs() < 1.0e-5);
        assert!((stable_previous_uv[1] - current_uv[1]).abs() < 1.0e-5);
    }

    #[test]
    fn skipped_frame_ids_continue_history_but_stale_ids_reset_it() {
        let camera = CameraSnapshot::default();
        let aspect = 16.0 / 9.0;
        let previous = PreviousFrame {
            frame_id: 10,
            scene: 1,
            surface_generation: 1,
            camera,
            aspect,
            view_projection: camera.view_projection(aspect),
            jitter_pixels: [0.0; 2],
            aa_blend: 0.78,
            volumetric_input: false,
            volumetric_signature: 0,
            light_direction: [0.0, -1.0, 0.0],
            light_intensity: 1.0,
            light_color: [1.0; 3],
        };
        let skipped_forward = snapshot_with_frame_id(12, camera);
        let duplicate = snapshot_with_frame_id(10, camera);
        let stale = snapshot_with_frame_id(9, camera);

        assert!(!temporal_reprojection_history_discontinuous(
            previous,
            &skipped_forward,
            camera,
            aspect,
        ));
        assert!(temporal_reprojection_history_discontinuous(
            previous, &duplicate, camera, aspect,
        ));
        assert!(temporal_reprojection_history_discontinuous(
            previous, &stale, camera, aspect,
        ));
    }

    #[test]
    fn volumetric_history_is_static_camera_only_and_tracks_medium_signature() {
        let camera = CameraSnapshot::default();
        let aspect = 16.0 / 9.0;
        let previous = PreviousFrame {
            frame_id: 10,
            scene: 1,
            surface_generation: 1,
            camera,
            aspect,
            view_projection: camera.view_projection(aspect),
            jitter_pixels: [0.0; 2],
            aa_blend: 0.78,
            volumetric_input: true,
            volumetric_signature: 41,
            light_direction: [0.0, -1.0, 0.0],
            light_intensity: 1.0,
            light_color: [1.0; 3],
        };
        let stable_snapshot = snapshot_with_frame_id(11, camera);
        assert!(!temporal_reprojection_history_discontinuous(
            previous,
            &stable_snapshot,
            camera,
            aspect,
        ));
        assert!(!volumetric_history_discontinuous(
            previous,
            &stable_snapshot,
            camera,
            aspect,
            41,
        ));

        let moved_camera = CameraSnapshot {
            eye: [camera.eye[0] + 0.01, camera.eye[1], camera.eye[2]],
            ..camera
        };
        let moved_snapshot = snapshot_with_frame_id(11, moved_camera);
        assert!(!volumetric_history_discontinuous(
            previous,
            &moved_snapshot,
            moved_camera,
            aspect,
            41,
        ));
        let cut_camera = CameraSnapshot {
            eye: [camera.eye[0] + 10.0, camera.eye[1], camera.eye[2]],
            ..camera
        };
        let cut_snapshot = snapshot_with_frame_id(11, cut_camera);
        assert!(volumetric_history_discontinuous(
            previous,
            &cut_snapshot,
            cut_camera,
            aspect,
            41,
        ));
        assert!(volumetric_history_discontinuous(
            previous,
            &stable_snapshot,
            camera,
            aspect,
            42,
        ));
    }

    #[test]
    fn volumetric_light_response_preserves_sub_milliradian_sun_motion() {
        let previous = PreviousFrame {
            frame_id: 10,
            scene: 1,
            surface_generation: 1,
            camera: CameraSnapshot::default(),
            aspect: 16.0 / 9.0,
            view_projection: CameraSnapshot::default().view_projection(16.0 / 9.0),
            jitter_pixels: [0.0; 2],
            aa_blend: 0.78,
            volumetric_input: true,
            volumetric_signature: 0,
            light_direction: [0.0, -1.0, 0.0],
            light_intensity: 1.0,
            light_color: [1.0; 3],
        };
        // The dot product of these directions rounds to one in f32.  The cross-product/atan2
        // implementation must still report the 0.1 mrad motion so a moving CSM silhouette gets
        // current-frame weight instead of accumulating as a ghost box.
        let current_direction = [0.0001, -1.0, 0.0];
        let signal = volumetric_light_change(current_direction, 1.0, [1.0; 3], previous);
        assert!(
            signal > 0.15,
            "sub-milliradian light motion was lost: {signal}"
        );
    }

    #[test]
    fn volumetric_light_motion_does_not_reset_history_signature() {
        let camera = CameraSnapshot::default();
        let mut current_snapshot = snapshot_with_frame_id(11, camera);
        current_snapshot.lights.push(
            LightPacket::new(1.15).with_direction_and_color([0.001, -1.0, 0.0], [0.95, 1.05, 1.15]),
        );
        let quality = RenderQualitySettings::high_quality();
        let signature = volumetric_history_signature(quality, true);
        let previous = PreviousFrame {
            frame_id: 10,
            scene: 1,
            surface_generation: 1,
            camera,
            aspect: 16.0 / 9.0,
            view_projection: camera.view_projection(16.0 / 9.0),
            jitter_pixels: [0.0; 2],
            aa_blend: 0.78,
            volumetric_input: true,
            volumetric_signature: signature,
            light_direction: [0.0, -1.0, 0.0],
            light_intensity: 1.0,
            light_color: [1.0; 3],
        };
        assert!(
            !volumetric_history_discontinuous(
                previous,
                &current_snapshot,
                camera,
                16.0 / 9.0,
                signature,
            ),
            "continuous Sun motion must be handled by the light response, not a full history reset"
        );
        assert!(!directional_light_history_cut(
            [0.001, -1.0, 0.0],
            1.15,
            [0.95, 1.05, 1.15],
            Some(previous),
        ));
    }

    #[test]
    fn directional_light_cut_requires_a_real_light_replacement() {
        let previous = PreviousFrame {
            frame_id: 10,
            scene: 1,
            surface_generation: 1,
            camera: CameraSnapshot::default(),
            aspect: 16.0 / 9.0,
            view_projection: CameraSnapshot::default().view_projection(16.0 / 9.0),
            jitter_pixels: [0.0; 2],
            aa_blend: 0.78,
            volumetric_input: true,
            volumetric_signature: 0,
            light_direction: [0.0, -1.0, 0.0],
            light_intensity: 1.0,
            light_color: [1.0; 3],
        };
        assert!(directional_light_history_cut(
            [1.0, 0.0, 0.0],
            1.0,
            [1.0; 3],
            Some(previous),
        ));
        assert!(directional_light_history_cut(
            [0.0, -1.0, 0.0],
            4.0,
            [1.0; 3],
            Some(previous),
        ));
        assert!(!directional_light_history_cut(
            [0.0004, -1.0, 0.0],
            1.02,
            [1.02, 0.99, 1.0],
            Some(previous),
        ));
    }

    #[test]
    fn temporal_effect_state_commits_continuous_frames_and_resets_after_invalidation() {
        let camera = CameraSnapshot::default();
        let extent = vk::Extent2D {
            width: 1600,
            height: 900,
        };
        let light = LightPacket::new(1.0);
        let mut state = TemporalEffectsState::default();

        let first_snapshot = snapshot_with_frame_id(1, camera);
        let first = state.prepare(&first_snapshot, camera, extent, light);
        assert!(first.reset);
        assert_eq!(first.sun_feedback, 0.0);
        assert_eq!(first.pcss_feedback, 0.0);
        state.commit();

        let mut second_snapshot = snapshot_with_frame_id(2, camera);
        second_snapshot.frame_rate_hz = 60.0;
        let second = state.prepare(&second_snapshot, camera, extent, light);
        assert!(!second.reset);
        assert_eq!(
            second.previous_view_projection,
            camera.view_projection(extent.width as f32 / extent.height as f32)
        );
        assert!(second.sun_feedback > second.pcss_feedback);
        assert!(second.pcss_feedback > 0.0 && second.sun_feedback < 1.0);
        assert!(second.sample_phase > 0.0 && second.sample_phase < 1.0);
        assert_ne!(first.sample_phase, second.sample_phase);

        state.invalidate();
        state.commit();
        let third_snapshot = snapshot_with_frame_id(3, camera);
        let third = state.prepare(&third_snapshot, camera, extent, light);
        assert!(third.reset);
        assert_eq!(third.sun_feedback, 0.0);
        assert_eq!(third.pcss_feedback, 0.0);
    }

    #[test]
    fn temporal_sample_phase_is_bounded_and_changes_across_the_history_horizon() {
        let light = LightPacket::new(1.0);
        let phases: Vec<_> = (1..=TEMPORAL_PHASE_SAMPLE_COUNT)
            .map(|frame_id| temporal_sample_phase(frame_id, light, 0.0))
            .collect();
        assert!(phases.iter().all(|phase| *phase > 0.0 && *phase < 1.0));
        assert!(phases.windows(2).any(|window| window[0] != window[1]));
        assert!(
            phases
                .iter()
                .map(|phase| phase.to_bits())
                .collect::<std::collections::HashSet<_>>()
                .len()
                > 32
        );
        assert_eq!(
            temporal_sample_phase(1, light, 0.0),
            temporal_sample_phase(1 + TEMPORAL_PHASE_SAMPLE_COUNT, light, 0.0),
            "the bounded phase repeats after the independent 64-frame effect horizon"
        );
    }

    #[test]
    fn temporal_phase_and_light_motion_follow_a_moving_sun() {
        let camera = CameraSnapshot::default();
        let extent = vk::Extent2D {
            width: 1600,
            height: 900,
        };
        let static_light =
            LightPacket::new(1.0).with_direction_and_color([0.0, -1.0, 0.0], [1.0; 3]);
        let moved_light = static_light.with_direction_and_color([0.0001, -1.0, 0.0], [1.0; 3]);
        let mut state = TemporalEffectsState::default();
        let first_snapshot = snapshot_with_frame_id(1, camera);
        let first = state.prepare(&first_snapshot, camera, extent, static_light);
        state.commit();

        let second_snapshot = snapshot_with_frame_id(2, camera);
        let second = state.prepare(&second_snapshot, camera, extent, moved_light);
        assert!(second.light_motion > 0.15);
        assert_ne!(
            temporal_sample_phase(8, static_light, 0.0),
            temporal_sample_phase(8, moved_light, second.light_motion),
            "the same frame must produce a different phase when the Sun moves"
        );
        assert_ne!(
            first.sample_phase, second.sample_phase,
            "the effect phase must include moving-light state, not only the frame id"
        );
    }

    #[test]
    fn pcss_camera_motion_reactivity_tracks_view_rotation_without_a_history_cut() {
        let camera = CameraSnapshot::default();
        let rotated_camera = CameraSnapshot {
            target: [camera.target[0] + 0.005, camera.target[1], camera.target[2]],
            ..camera
        };
        let extent = vk::Extent2D {
            width: 1600,
            height: 900,
        };
        let light = LightPacket::new(1.0);
        let mut state = TemporalEffectsState::default();

        let first_snapshot = snapshot_with_frame_id(1, camera);
        state.prepare(&first_snapshot, camera, extent, light);
        state.commit();

        let second_snapshot = snapshot_with_frame_id(2, rotated_camera);
        let second = state.prepare(&second_snapshot, rotated_camera, extent, light);
        assert!(
            !second.reset,
            "a normal view rotation must not discard all history"
        );
        assert!(
            second.pcss_camera_motion > 0.0 && second.pcss_camera_motion < 1.0,
            "PCSS must shorten history continuously for ordinary camera motion"
        );
    }

    #[test]
    fn pcss_reactivity_keeps_temporal_accumulation_during_light_motion() {
        let light_only = TemporalEffectFrame {
            light_motion: 1.0,
            ..TemporalEffectFrame::default()
        };
        assert_eq!(light_only.pcss_reactivity(), PCSS_LIGHT_REACTIVITY_SCALE);

        let camera_motion = TemporalEffectFrame {
            light_motion: 0.25,
            pcss_camera_motion: 0.8,
            ..TemporalEffectFrame::default()
        };
        assert_eq!(camera_motion.pcss_reactivity(), 0.8);
    }

    #[test]
    fn temporal_feedback_uses_a_slow_frame_rate_invariant_decay() {
        let at_60_hz = exponential_history_feedback(0.78, false, 60.0);
        let at_120_hz = exponential_history_feedback(0.78, false, 120.0);
        let at_240_hz = exponential_history_feedback(0.78, false, 240.0);

        // A lower presentation rate advances the same time-based filter by a larger amount per
        // frame, while a higher rate takes smaller per-frame steps.  The retained history after
        // one second must remain approximately identical.
        assert!(at_60_hz < at_120_hz && at_120_hz < at_240_hz);
        let retained_at_60_hz = at_60_hz.powi(60);
        let retained_at_120_hz = at_120_hz.powi(120);
        let retained_at_240_hz = at_240_hz.powi(240);
        assert!((retained_at_60_hz - retained_at_120_hz).abs() < 0.0001);
        assert!((retained_at_120_hz - retained_at_240_hz).abs() < 0.0001);

        let half_life_seconds = -TAA_LN_TWO / (at_120_hz.ln() * 120.0);
        assert!(
            half_life_seconds >= TAA_MIN_HALF_LIFE_SECONDS - 0.0001,
            "half-life is shorter than the jitter-period floor: {half_life_seconds}"
        );

        // The configured time scale is intentionally slower than the old 120 Hz profile value.
        assert!(at_120_hz > 0.80 + 0.16 * 0.78);

        let sun_at_120_hz = temporal_effect_feedback(120.0, 0.955);
        let sun_at_60_hz = temporal_effect_feedback(60.0, 0.955);
        let sun_at_240_hz = temporal_effect_feedback(240.0, 0.955);
        assert!((sun_at_120_hz - 0.955).abs() < 0.0001);
        assert!(sun_at_60_hz < sun_at_120_hz && sun_at_120_hz < sun_at_240_hz);
        let sun_retained_at_60_hz = sun_at_60_hz.powi(60);
        let sun_retained_at_120_hz = sun_at_120_hz.powi(120);
        let sun_retained_at_240_hz = sun_at_240_hz.powi(240);
        assert!((sun_retained_at_60_hz - sun_retained_at_120_hz).abs() < 0.0001);
        assert!((sun_retained_at_120_hz - sun_retained_at_240_hz).abs() < 0.0001);
    }

    #[test]
    fn temporal_uniform_layout_matches_slang_constant_buffer() {
        assert_eq!(size_of::<TemporalResolveUniform>(), 384);
        assert_eq!(
            std::mem::offset_of!(TemporalResolveUniform, inverse_current_view),
            192
        );
        assert_eq!(
            std::mem::offset_of!(TemporalResolveUniform, inverse_previous_view),
            256
        );
        assert_eq!(
            std::mem::offset_of!(TemporalResolveUniform, texel_feedback_reset),
            320
        );
        assert_eq!(std::mem::offset_of!(TemporalResolveUniform, effects), 368);
    }

    #[test]
    fn descriptors_cover_current_color_taa_and_frame_ping_pong() {
        let frame_slots = 2;
        let mut indices = std::collections::BTreeSet::new();
        for current_color_input in 0..CURRENT_COLOR_INPUT_COUNT {
            for taa_write in 0..TAA_HISTORY_COUNT {
                for slot in 0..frame_slots {
                    indices.insert(taa_descriptor_index(
                        current_color_input,
                        taa_write,
                        slot,
                        frame_slots,
                    ));
                }
            }
        }
        assert_eq!(
            indices,
            (0..CURRENT_COLOR_INPUT_COUNT * TAA_HISTORY_COUNT * frame_slots).collect()
        );
    }
}
