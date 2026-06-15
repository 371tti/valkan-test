use std::collections::{BTreeMap, BTreeSet, VecDeque};

use thiserror::Error;

const SCENE_PASS: &str = "scene";
const POST_PASS: &str = "post";
const FRAMEBUFFER_READBACK_PASS: &str = "framebuffer_readback";
const PRESENT_PASS: &str = "present";
const SHADOW_PASSES: [&str; SHADOW_CASCADE_COUNT] = [
    "shadow_cascade_0",
    "shadow_cascade_1",
    "shadow_cascade_2",
    "shadow_cascade_3",
];
const TRANSLUCENT_SHADOW_PASSES: [&str; SHADOW_CASCADE_COUNT] = [
    "translucent_shadow_0",
    "translucent_shadow_1",
    "translucent_shadow_2",
    "translucent_shadow_3",
];

pub(crate) const SHADOW_CASCADE_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum GraphResource {
    SwapchainImage,
    ShadowCascade0,
    ShadowCascade1,
    ShadowCascade2,
    ShadowCascade3,
    TranslucentShadow0,
    TranslucentShadow1,
    TranslucentShadow2,
    TranslucentShadow3,
    SceneColor,
    SceneNormalRoughness,
    SceneTransparentNormalRoughness,
    SceneDepth,
}

impl GraphResource {
    /// Returns the stable resource name used in graph logs and diagnostics.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::SwapchainImage => "swapchain_image",
            Self::ShadowCascade0 => "shadow_cascade_0",
            Self::ShadowCascade1 => "shadow_cascade_1",
            Self::ShadowCascade2 => "shadow_cascade_2",
            Self::ShadowCascade3 => "shadow_cascade_3",
            Self::TranslucentShadow0 => "translucent_shadow_0",
            Self::TranslucentShadow1 => "translucent_shadow_1",
            Self::TranslucentShadow2 => "translucent_shadow_2",
            Self::TranslucentShadow3 => "translucent_shadow_3",
            Self::SceneColor => "scene_color",
            Self::SceneNormalRoughness => "scene_normal_roughness",
            Self::SceneTransparentNormalRoughness => "scene_transparent_normal_roughness",
            Self::SceneDepth => "scene_depth",
        }
    }
}

pub(crate) const SHADOW_CASCADE_RESOURCES: [GraphResource; SHADOW_CASCADE_COUNT] = [
    GraphResource::ShadowCascade0,
    GraphResource::ShadowCascade1,
    GraphResource::ShadowCascade2,
    GraphResource::ShadowCascade3,
];

pub(crate) const TRANSLUCENT_SHADOW_RESOURCES: [GraphResource; SHADOW_CASCADE_COUNT] = [
    GraphResource::TranslucentShadow0,
    GraphResource::TranslucentShadow1,
    GraphResource::TranslucentShadow2,
    GraphResource::TranslucentShadow3,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResourceState {
    Undefined,
    ColorAttachment,
    DepthAttachment,
    ShaderRead,
    TransferSrc,
    Present,
}

impl ResourceState {
    /// Returns the compact state name used when tracing graph transitions.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Undefined => "undefined",
            Self::ColorAttachment => "color_attachment",
            Self::DepthAttachment => "depth_attachment",
            Self::ShaderRead => "shader_read",
            Self::TransferSrc => "transfer_src",
            Self::Present => "present",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoadOp {
    Clear,
    Load,
}

impl LoadOp {
    /// Returns the compact load operation name used in graph trace logs.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Clear => "clear",
            Self::Load => "load",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StoreOp {
    Store,
}

impl StoreOp {
    /// Returns the compact store operation name used in graph trace logs.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Store => "store",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GraphResourceDecl {
    resource: GraphResource,
    initial: ResourceState,
    final_state: ResourceState,
}

impl GraphResourceDecl {
    /// Creates a graph resource declaration with its frame-local state contract.
    pub(crate) fn new(
        resource: GraphResource,
        initial: ResourceState,
        final_state: ResourceState,
    ) -> Self {
        Self {
            resource,
            initial,
            final_state,
        }
    }

    /// Returns the declared resource handle.
    pub(crate) fn resource(&self) -> GraphResource {
        self.resource
    }

    /// Returns the state expected before the first pass touches this resource.
    pub(crate) fn initial(&self) -> ResourceState {
        self.initial
    }

    /// Returns the state required after the final pass finishes this resource.
    pub(crate) fn final_state(&self) -> ResourceState {
        self.final_state
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PassInput {
    resource: GraphResource,
    state: ResourceState,
}

impl PassInput {
    /// Declares that a pass reads one resource in the requested state.
    pub(crate) fn read(resource: GraphResource, state: ResourceState) -> Self {
        Self { resource, state }
    }

    /// Returns the resource read by this pass input.
    pub(crate) fn resource(&self) -> GraphResource {
        self.resource
    }

    /// Returns the state required while this pass reads the resource.
    pub(crate) fn state(&self) -> ResourceState {
        self.state
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PassOutput {
    resource: GraphResource,
    state: ResourceState,
    load: LoadOp,
    store: StoreOp,
    clear_color: [f32; 4],
}

impl PassOutput {
    /// Creates one color output written by a graph pass.
    pub(crate) fn color_clear(resource: GraphResource, clear_color: [f32; 4]) -> Self {
        Self {
            resource,
            state: ResourceState::ColorAttachment,
            load: LoadOp::Clear,
            store: StoreOp::Store,
            clear_color,
        }
    }

    /// Creates one color output that preserves its prior contents before writing.
    pub(crate) fn color_load(resource: GraphResource) -> Self {
        Self {
            resource,
            state: ResourceState::ColorAttachment,
            load: LoadOp::Load,
            store: StoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Creates one depth output whose contents must survive for later shader reads.
    pub(crate) fn depth_clear_store(resource: GraphResource) -> Self {
        Self {
            resource,
            state: ResourceState::DepthAttachment,
            load: LoadOp::Clear,
            store: StoreOp::Store,
            clear_color: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Returns the resource written by this pass output.
    pub(crate) fn resource(&self) -> GraphResource {
        self.resource
    }

    /// Returns the resource state required while this output is written.
    pub(crate) fn state(&self) -> ResourceState {
        self.state
    }

    /// Returns the load policy for this output.
    pub(crate) fn load(&self) -> LoadOp {
        self.load
    }

    /// Returns the store policy for this output.
    pub(crate) fn store(&self) -> StoreOp {
        self.store
    }

    /// Returns the clear color copied into the backend render pass.
    pub(crate) fn clear_color(&self) -> [f32; 4] {
        self.clear_color
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphPass {
    name: &'static str,
    reads: Vec<PassInput>,
    writes: Vec<PassOutput>,
    side_effect: bool,
}

impl GraphPass {
    /// Creates an empty pass declaration that can be extended with reads and writes.
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            reads: Vec::new(),
            writes: Vec::new(),
            side_effect: false,
        }
    }

    /// Adds one resource read to this pass declaration.
    pub(crate) fn read(mut self, input: PassInput) -> Self {
        self.reads.push(input);
        self
    }

    /// Adds one resource write to this pass declaration.
    pub(crate) fn write(mut self, output: PassOutput) -> Self {
        self.writes.push(output);
        self
    }

    /// Marks the pass as externally visible so culling keeps it even without graph outputs.
    pub(crate) fn with_side_effect(mut self) -> Self {
        self.side_effect = true;
        self
    }

    /// Returns the stable pass name used for logs and future tooling.
    pub(crate) fn name(&self) -> &'static str {
        self.name
    }

    /// Returns all resource reads declared by this pass.
    pub(crate) fn reads(&self) -> &[PassInput] {
        &self.reads
    }

    /// Returns all resource writes declared by this pass.
    pub(crate) fn writes(&self) -> &[PassOutput] {
        &self.writes
    }

    /// Returns whether this pass has an externally visible side effect.
    fn side_effect(&self) -> bool {
        self.side_effect
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum BarrierLocation {
    BeforePass(&'static str),
    AfterGraph,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResourceBarrier {
    resource: GraphResource,
    from: ResourceState,
    to: ResourceState,
    location: BarrierLocation,
}

impl ResourceBarrier {
    /// Creates one explicit resource state step emitted by the graph plan.
    fn new(
        resource: GraphResource,
        from: ResourceState,
        to: ResourceState,
        location: BarrierLocation,
    ) -> Self {
        Self {
            resource,
            from,
            to,
            location,
        }
    }

    /// Returns the resource whose state is changed by this barrier.
    pub(crate) fn resource(&self) -> GraphResource {
        self.resource
    }

    /// Returns the resource state before this barrier executes.
    pub(crate) fn from(&self) -> ResourceState {
        self.from
    }

    /// Returns the resource state after this barrier executes.
    pub(crate) fn to(&self) -> ResourceState {
        self.to
    }

    /// Returns where command recording should emit this barrier.
    pub(crate) fn location(&self) -> BarrierLocation {
        self.location
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResourceLifetime {
    resource: GraphResource,
    first_pass: usize,
    last_pass: usize,
}

impl ResourceLifetime {
    /// Creates the inclusive pass-index range where a resource is used.
    fn new(resource: GraphResource, first_pass: usize, last_pass: usize) -> Self {
        Self {
            resource,
            first_pass,
            last_pass,
        }
    }

    /// Returns the resource whose lifetime is described.
    pub(crate) fn resource(&self) -> GraphResource {
        self.resource
    }

    /// Returns the first compiled pass index that touches this resource.
    pub(crate) fn first_pass(&self) -> usize {
        self.first_pass
    }

    /// Returns the last compiled pass index that touches this resource.
    pub(crate) fn last_pass(&self) -> usize {
        self.last_pass
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TransientAliasCandidate {
    first: GraphResource,
    second: GraphResource,
}

impl TransientAliasCandidate {
    /// Creates one non-overlapping resource pair that a future allocator can alias.
    fn new(first: GraphResource, second: GraphResource) -> Self {
        Self { first, second }
    }

    /// Returns the first resource in the candidate pair.
    pub(crate) fn first(&self) -> GraphResource {
        self.first
    }

    /// Returns the second resource in the candidate pair.
    pub(crate) fn second(&self) -> GraphResource {
        self.second
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GraphOptimizationHints {
    barrier_merge_candidates: usize,
    transient_alias_candidates: Vec<TransientAliasCandidate>,
    render_pass_merge_candidates: usize,
}

impl GraphOptimizationHints {
    /// Returns adjacent barrier groups that a future backend can merge.
    pub(crate) fn barrier_merge_candidates(&self) -> usize {
        self.barrier_merge_candidates
    }

    /// Returns resource lifetime pairs that do not overlap.
    pub(crate) fn transient_alias_candidates(&self) -> &[TransientAliasCandidate] {
        &self.transient_alias_candidates
    }

    /// Returns adjacent pass pairs that may become one backend render pass.
    pub(crate) fn render_pass_merge_candidates(&self) -> usize {
        self.render_pass_merge_candidates
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrameGraphPlan {
    resources: Vec<GraphResourceDecl>,
    passes: Vec<GraphPass>,
    barriers: Vec<ResourceBarrier>,
    barriers_by_location: BTreeMap<BarrierLocation, Vec<ResourceBarrier>>,
    lifetimes: Vec<ResourceLifetime>,
    hints: GraphOptimizationHints,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrameGraphInitialStates {
    swapchain_image: ResourceState,
    shadow_cascades: [ResourceState; SHADOW_CASCADE_COUNT],
    translucent_shadows: [ResourceState; SHADOW_CASCADE_COUNT],
    scene_color: ResourceState,
    scene_normal_roughness: ResourceState,
    scene_transparent_normal_roughness: ResourceState,
    scene_depth: ResourceState,
}

impl FrameGraphInitialStates {
    /// Captures the actual image layouts known by the Vulkan executor before graph compilation.
    pub(crate) fn new(
        swapchain_image: ResourceState,
        shadow_cascades: [ResourceState; SHADOW_CASCADE_COUNT],
        translucent_shadows: [ResourceState; SHADOW_CASCADE_COUNT],
        scene_color: ResourceState,
        scene_normal_roughness: ResourceState,
        scene_transparent_normal_roughness: ResourceState,
        scene_depth: ResourceState,
    ) -> Self {
        Self {
            swapchain_image,
            shadow_cascades,
            translucent_shadows,
            scene_color,
            scene_normal_roughness,
            scene_transparent_normal_roughness,
            scene_depth,
        }
    }

    /// Returns the current state of the acquired swapchain image.
    pub(crate) fn swapchain_image(self) -> ResourceState {
        self.swapchain_image
    }

    /// Returns the current state of every persistent shadow cascade depth map.
    pub(crate) fn shadow_cascades(self) -> [ResourceState; SHADOW_CASCADE_COUNT] {
        self.shadow_cascades
    }

    /// Returns the current state of every persistent translucent shadow transmittance map.
    pub(crate) fn translucent_shadows(self) -> [ResourceState; SHADOW_CASCADE_COUNT] {
        self.translucent_shadows
    }

    /// Returns the current state of the persistent scene color target.
    pub(crate) fn scene_color(self) -> ResourceState {
        self.scene_color
    }

    /// Returns the current state of the persistent scene normal/roughness target.
    pub(crate) fn scene_normal_roughness(self) -> ResourceState {
        self.scene_normal_roughness
    }

    /// Returns the current state of the persistent transparent material metadata target.
    pub(crate) fn scene_transparent_normal_roughness(self) -> ResourceState {
        self.scene_transparent_normal_roughness
    }

    /// Returns the current state of the persistent scene depth target.
    pub(crate) fn scene_depth(self) -> ResourceState {
        self.scene_depth
    }
}

impl FrameGraphPlan {
    /// Builds the executable frame graph, optionally exposing the final image for readback.
    pub(crate) fn standard_frame_with_readback(
        clear_color: [f32; 4],
        initial: FrameGraphInitialStates,
        framebuffer_readback: bool,
        shadow_casters: bool,
        translucent_shadows: bool,
    ) -> Result<Self, GraphCompileError> {
        Self::standard_frame_with_shadow_refresh(
            clear_color,
            initial,
            framebuffer_readback,
            shadow_casters,
            translucent_shadows,
            true,
        )
    }

    /// Builds the executable frame graph while optionally reusing cached shadow maps.
    pub(crate) fn standard_frame_with_shadow_refresh(
        clear_color: [f32; 4],
        initial: FrameGraphInitialStates,
        framebuffer_readback: bool,
        shadow_casters: bool,
        translucent_shadows: bool,
        refresh_shadows: bool,
    ) -> Result<Self, GraphCompileError> {
        let mut builder = FrameGraphBuilder::new();
        if shadow_casters {
            for (resource, state) in SHADOW_CASCADE_RESOURCES
                .into_iter()
                .zip(initial.shadow_cascades())
            {
                builder = builder.resource(GraphResourceDecl::new(
                    resource,
                    state,
                    ResourceState::ShaderRead,
                ));
            }
        }
        if shadow_casters && translucent_shadows {
            for (resource, state) in TRANSLUCENT_SHADOW_RESOURCES
                .into_iter()
                .zip(initial.translucent_shadows())
            {
                builder = builder.resource(GraphResourceDecl::new(
                    resource,
                    state,
                    ResourceState::ShaderRead,
                ));
            }
        }

        let mut builder = builder
            .resource(GraphResourceDecl::new(
                GraphResource::SceneColor,
                initial.scene_color(),
                ResourceState::ShaderRead,
            ))
            .resource(GraphResourceDecl::new(
                GraphResource::SceneNormalRoughness,
                initial.scene_normal_roughness(),
                ResourceState::ShaderRead,
            ))
            .resource(GraphResourceDecl::new(
                GraphResource::SceneTransparentNormalRoughness,
                initial.scene_transparent_normal_roughness(),
                ResourceState::ShaderRead,
            ))
            .resource(GraphResourceDecl::new(
                GraphResource::SceneDepth,
                initial.scene_depth(),
                ResourceState::ShaderRead,
            ))
            .resource(GraphResourceDecl::new(
                GraphResource::SwapchainImage,
                initial.swapchain_image(),
                ResourceState::Present,
            ));

        if shadow_casters && refresh_shadows {
            for (pass_name, resource) in SHADOW_PASSES.into_iter().zip(SHADOW_CASCADE_RESOURCES) {
                builder = builder
                    .pass(GraphPass::new(pass_name).write(PassOutput::depth_clear_store(resource)));
            }
        }
        if shadow_casters && translucent_shadows && refresh_shadows {
            for (pass_name, color_resource) in TRANSLUCENT_SHADOW_PASSES
                .into_iter()
                .zip(TRANSLUCENT_SHADOW_RESOURCES)
            {
                let mut pass = GraphPass::new(pass_name);
                for sampled_depth_resource in SHADOW_CASCADE_RESOURCES {
                    pass = pass.read(PassInput::read(
                        sampled_depth_resource,
                        ResourceState::ShaderRead,
                    ));
                }
                builder = builder.pass(pass.write(PassOutput::color_clear(
                    color_resource,
                    [1.0, 1.0, 1.0, 1.0],
                )));
            }
        }

        let mut scene_pass = GraphPass::new(SCENE_PASS);
        if shadow_casters {
            for resource in SHADOW_CASCADE_RESOURCES {
                scene_pass = scene_pass.read(PassInput::read(resource, ResourceState::ShaderRead));
            }
            if translucent_shadows {
                for resource in TRANSLUCENT_SHADOW_RESOURCES {
                    scene_pass =
                        scene_pass.read(PassInput::read(resource, ResourceState::ShaderRead));
                }
            }
        }
        builder = builder
            .pass(
                scene_pass
                    .write(PassOutput::color_clear(
                        GraphResource::SceneColor,
                        clear_color,
                    ))
                    .write(PassOutput::color_clear(
                        GraphResource::SceneNormalRoughness,
                        [0.5, 0.5, 1.0, 0.0],
                    ))
                    .write(PassOutput::color_clear(
                        GraphResource::SceneTransparentNormalRoughness,
                        [0.0, 0.0, 0.0, 0.0],
                    ))
                    .write(PassOutput::depth_clear_store(GraphResource::SceneDepth)),
            )
            .pass(
                GraphPass::new(POST_PASS)
                    .read(PassInput::read(
                        GraphResource::SceneColor,
                        ResourceState::ShaderRead,
                    ))
                    .read(PassInput::read(
                        GraphResource::SceneNormalRoughness,
                        ResourceState::ShaderRead,
                    ))
                    .read(PassInput::read(
                        GraphResource::SceneTransparentNormalRoughness,
                        ResourceState::ShaderRead,
                    ))
                    .read(PassInput::read(
                        GraphResource::SceneDepth,
                        ResourceState::ShaderRead,
                    ))
                    .write(PassOutput::color_load(GraphResource::SwapchainImage)),
            );

        if framebuffer_readback {
            builder = builder.pass(
                GraphPass::new(FRAMEBUFFER_READBACK_PASS)
                    .read(PassInput::read(
                        GraphResource::SwapchainImage,
                        ResourceState::TransferSrc,
                    ))
                    .with_side_effect(),
            );
        }

        builder
            .pass(
                GraphPass::new(PRESENT_PASS)
                    .read(PassInput::read(
                        GraphResource::SwapchainImage,
                        ResourceState::Present,
                    ))
                    .with_side_effect(),
            )
            .compile()
    }

    /// Returns the number of resources declared by this compiled graph.
    pub(crate) fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Returns the number of executable passes retained after culling.
    pub(crate) fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Returns the number of resources with at least one state transition.
    pub(crate) fn transition_count(&self) -> usize {
        self.barriers
            .iter()
            .map(ResourceBarrier::resource)
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Returns the number of concrete barriers emitted by this compiled graph.
    pub(crate) fn barrier_count(&self) -> usize {
        self.barriers.len()
    }

    /// Returns all declared resources in deterministic order.
    pub(crate) fn resources(&self) -> &[GraphResourceDecl] {
        &self.resources
    }

    /// Returns the executable passes after dependency sorting and culling.
    pub(crate) fn passes(&self) -> &[GraphPass] {
        &self.passes
    }

    /// Returns the concrete barriers emitted around compiled passes.
    #[cfg(test)]
    pub(crate) fn barriers(&self) -> &[ResourceBarrier] {
        &self.barriers
    }

    /// Returns barriers that must be recorded at one graph location.
    pub(crate) fn barriers_at(&self, location: BarrierLocation) -> &[ResourceBarrier] {
        self.barriers_by_location
            .get(&location)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns resource lifetimes in the compiled pass order.
    pub(crate) fn lifetimes(&self) -> &[ResourceLifetime] {
        &self.lifetimes
    }

    /// Returns optimization metadata produced without changing backend execution.
    pub(crate) fn optimization_hints(&self) -> &GraphOptimizationHints {
        &self.hints
    }

    /// Returns the declared final state for a compiled graph resource.
    pub(crate) fn final_state_for(&self, resource: GraphResource) -> Option<ResourceState> {
        self.resources
            .iter()
            .find(|decl| decl.resource() == resource)
            .map(GraphResourceDecl::final_state)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct FrameGraphBuilder {
    resources: Vec<GraphResourceDecl>,
    passes: Vec<GraphPass>,
}

impl FrameGraphBuilder {
    /// Creates an empty graph declaration before resources and passes are added.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Adds one frame-local resource declaration to the graph.
    pub(crate) fn resource(mut self, resource: GraphResourceDecl) -> Self {
        self.resources.push(resource);
        self
    }

    /// Adds one pass declaration to the graph.
    pub(crate) fn pass(mut self, pass: GraphPass) -> Self {
        self.passes.push(pass);
        self
    }

    /// Validates, culls, orders, and plans barriers for the declared graph.
    pub(crate) fn compile(self) -> Result<FrameGraphPlan, GraphCompileError> {
        validate_unique_resources(&self.resources)?;
        validate_unique_passes(&self.passes)?;
        validate_pass_resources(&self.resources, &self.passes)?;

        let kept = cull_passes(&self.resources, &self.passes);
        let ordered = order_passes(&kept)?;
        let barriers = plan_barriers(&self.resources, &ordered);
        let barriers_by_location = group_barriers_by_location(&barriers);
        let lifetimes = compute_lifetimes(&ordered);
        let hints = compute_optimization_hints(&ordered, &barriers, &lifetimes);

        Ok(FrameGraphPlan {
            resources: self.resources,
            passes: ordered,
            barriers,
            barriers_by_location,
            lifetimes,
            hints,
        })
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub(crate) enum GraphCompileError {
    #[error("graph resource is declared more than once: {0}")]
    DuplicateResource(&'static str),
    #[error("graph pass is declared more than once: {0}")]
    DuplicatePass(&'static str),
    #[error("graph pass {pass} references missing resource {resource}")]
    MissingResource {
        pass: &'static str,
        resource: &'static str,
    },
    #[error("graph dependency cycle includes pass: {0}")]
    Cycle(&'static str),
}

/// Rejects duplicated resource declarations before dependency planning begins.
fn validate_unique_resources(resources: &[GraphResourceDecl]) -> Result<(), GraphCompileError> {
    let mut seen = BTreeSet::new();
    for resource in resources {
        if !seen.insert(resource.resource()) {
            return Err(GraphCompileError::DuplicateResource(
                resource.resource().name(),
            ));
        }
    }

    Ok(())
}

/// Rejects duplicated pass names so logs and dependency errors stay unambiguous.
fn validate_unique_passes(passes: &[GraphPass]) -> Result<(), GraphCompileError> {
    let mut seen = BTreeSet::new();
    for pass in passes {
        if !seen.insert(pass.name()) {
            return Err(GraphCompileError::DuplicatePass(pass.name()));
        }
    }

    Ok(())
}

/// Verifies that every pass usage points at a declared graph resource.
fn validate_pass_resources(
    resources: &[GraphResourceDecl],
    passes: &[GraphPass],
) -> Result<(), GraphCompileError> {
    let declared = resources
        .iter()
        .map(GraphResourceDecl::resource)
        .collect::<BTreeSet<_>>();
    for pass in passes {
        for usage in pass.reads() {
            require_resource(&declared, pass.name(), usage.resource())?;
        }
        for usage in pass.writes() {
            require_resource(&declared, pass.name(), usage.resource())?;
        }
    }

    Ok(())
}

/// Returns an error when one usage references a resource missing from declarations.
fn require_resource(
    declared: &BTreeSet<GraphResource>,
    pass: &'static str,
    resource: GraphResource,
) -> Result<(), GraphCompileError> {
    if declared.contains(&resource) {
        Ok(())
    } else {
        Err(GraphCompileError::MissingResource {
            pass,
            resource: resource.name(),
        })
    }
}

/// Removes passes whose outputs cannot reach a final resource or side effect.
fn cull_passes(_resources: &[GraphResourceDecl], passes: &[GraphPass]) -> Vec<GraphPass> {
    let mut needed = BTreeSet::new();
    let mut keep = vec![false; passes.len()];

    for (index, pass) in passes.iter().enumerate().rev() {
        let writes_needed = pass
            .writes()
            .iter()
            .any(|output| needed.contains(&output.resource()));
        if !pass.side_effect() && !writes_needed {
            continue;
        }

        keep[index] = true;
        for input in pass.reads() {
            needed.insert(input.resource());
        }
    }

    passes
        .iter()
        .cloned()
        .zip(keep)
        .filter_map(|(pass, keep)| keep.then_some(pass))
        .collect()
}

/// Orders passes by generated dependency edges while preserving declaration order when possible.
fn order_passes(passes: &[GraphPass]) -> Result<Vec<GraphPass>, GraphCompileError> {
    let edges = build_dependency_edges(passes);
    let mut incoming = incoming_edge_counts(passes.len(), &edges);
    let mut outgoing = outgoing_edges(passes.len(), &edges);
    let mut ready = incoming
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (*count == 0).then_some(index))
        .collect::<VecDeque<_>>();
    let mut ordered = Vec::with_capacity(passes.len());

    while let Some(index) = ready.pop_front() {
        ordered.push(index);
        for target in std::mem::take(&mut outgoing[index]) {
            incoming[target] -= 1;
            if incoming[target] == 0 {
                ready.push_back(target);
            }
        }
    }

    if ordered.len() != passes.len() {
        let blocked = incoming
            .iter()
            .enumerate()
            .find_map(|(index, count)| (*count > 0).then_some(passes[index].name()))
            .unwrap_or("unknown");
        return Err(GraphCompileError::Cycle(blocked));
    }

    Ok(ordered
        .into_iter()
        .map(|index| passes[index].clone())
        .collect())
}

/// Builds write/read/write hazard edges from pass resource usage.
fn build_dependency_edges(passes: &[GraphPass]) -> BTreeSet<(usize, usize)> {
    let mut edges = BTreeSet::new();
    let mut last_writer = BTreeMap::<GraphResource, usize>::new();
    let mut readers_since_write = BTreeMap::<GraphResource, Vec<usize>>::new();

    for (index, pass) in passes.iter().enumerate() {
        for input in pass.reads() {
            if let Some(writer) = last_writer.get(&input.resource()) {
                edges.insert((*writer, index));
            }
            readers_since_write
                .entry(input.resource())
                .or_default()
                .push(index);
        }

        for output in pass.writes() {
            if let Some(writer) = last_writer.insert(output.resource(), index) {
                edges.insert((writer, index));
            }
            if let Some(readers) = readers_since_write.remove(&output.resource()) {
                for reader in readers {
                    if reader != index {
                        edges.insert((reader, index));
                    }
                }
            }
        }
    }

    edges
}

/// Counts incoming dependency edges for each pass index.
fn incoming_edge_counts(pass_count: usize, edges: &BTreeSet<(usize, usize)>) -> Vec<usize> {
    let mut incoming = vec![0; pass_count];
    for &(_, target) in edges {
        incoming[target] += 1;
    }
    incoming
}

/// Groups outgoing dependency edges by source pass index.
fn outgoing_edges(pass_count: usize, edges: &BTreeSet<(usize, usize)>) -> Vec<Vec<usize>> {
    let mut outgoing = vec![Vec::new(); pass_count];
    for &(source, target) in edges {
        outgoing[source].push(target);
    }
    outgoing
}

/// Emits resource state barriers before passes and after graph execution.
fn plan_barriers(resources: &[GraphResourceDecl], passes: &[GraphPass]) -> Vec<ResourceBarrier> {
    let decls = resources
        .iter()
        .map(|resource| (resource.resource(), *resource))
        .collect::<BTreeMap<_, _>>();
    let used = used_resources(passes);
    let mut states = resources
        .iter()
        .map(|resource| (resource.resource(), resource.initial()))
        .collect::<BTreeMap<_, _>>();
    let mut barriers = Vec::new();

    for pass in passes {
        for input in pass.reads() {
            transition_usage(
                &mut states,
                &mut barriers,
                input.resource(),
                input.state(),
                BarrierLocation::BeforePass(pass.name()),
            );
        }
        for output in pass.writes() {
            transition_usage(
                &mut states,
                &mut barriers,
                output.resource(),
                output.state(),
                BarrierLocation::BeforePass(pass.name()),
            );
        }
    }

    for (resource, state) in states {
        if !used.contains(&resource) {
            continue;
        }
        let Some(decl) = decls.get(&resource) else {
            continue;
        };
        if state != decl.final_state() {
            barriers.push(ResourceBarrier::new(
                resource,
                state,
                decl.final_state(),
                BarrierLocation::AfterGraph,
            ));
        }
    }

    barriers
}

/// Groups planned barriers by command-recording location for cheap backend lookup.
fn group_barriers_by_location(
    barriers: &[ResourceBarrier],
) -> BTreeMap<BarrierLocation, Vec<ResourceBarrier>> {
    let mut grouped = BTreeMap::<BarrierLocation, Vec<ResourceBarrier>>::new();
    for barrier in barriers {
        grouped
            .entry(barrier.location())
            .or_default()
            .push(*barrier);
    }
    grouped
}

/// Returns every resource referenced by retained pass usages.
fn used_resources(passes: &[GraphPass]) -> BTreeSet<GraphResource> {
    passes
        .iter()
        .flat_map(|pass| {
            pass.reads()
                .iter()
                .map(PassInput::resource)
                .chain(pass.writes().iter().map(PassOutput::resource))
        })
        .collect()
}

/// Adds a barrier when a resource must enter a new state for one pass usage.
fn transition_usage(
    states: &mut BTreeMap<GraphResource, ResourceState>,
    barriers: &mut Vec<ResourceBarrier>,
    resource: GraphResource,
    next: ResourceState,
    location: BarrierLocation,
) {
    let current = states
        .get(&resource)
        .copied()
        .unwrap_or(ResourceState::Undefined);
    if current == next {
        return;
    }

    barriers.push(ResourceBarrier::new(resource, current, next, location));
    states.insert(resource, next);
}

/// Computes each resource's inclusive use range in compiled pass order.
fn compute_lifetimes(passes: &[GraphPass]) -> Vec<ResourceLifetime> {
    let mut ranges = BTreeMap::<GraphResource, (usize, usize)>::new();

    for (index, pass) in passes.iter().enumerate() {
        for resource in pass
            .reads()
            .iter()
            .map(PassInput::resource)
            .chain(pass.writes().iter().map(PassOutput::resource))
        {
            ranges
                .entry(resource)
                .and_modify(|range| range.1 = index)
                .or_insert((index, index));
        }
    }

    ranges
        .into_iter()
        .map(|(resource, (first, last))| ResourceLifetime::new(resource, first, last))
        .collect()
}

/// Produces optimization metadata without changing the current backend execution plan.
fn compute_optimization_hints(
    passes: &[GraphPass],
    barriers: &[ResourceBarrier],
    lifetimes: &[ResourceLifetime],
) -> GraphOptimizationHints {
    GraphOptimizationHints {
        barrier_merge_candidates: count_adjacent_barrier_groups(barriers),
        transient_alias_candidates: find_transient_alias_candidates(lifetimes),
        render_pass_merge_candidates: count_render_pass_merge_candidates(passes),
    }
}

/// Counts barriers recorded at the same graph location as one future merge candidate group.
fn count_adjacent_barrier_groups(barriers: &[ResourceBarrier]) -> usize {
    let mut groups = BTreeMap::<BarrierLocation, usize>::new();
    for barrier in barriers {
        *groups.entry(barrier.location()).or_default() += 1;
    }

    groups.values().filter(|&&count| count > 1).count()
}

/// Finds non-overlapping resource lifetime pairs for future transient allocation aliasing.
fn find_transient_alias_candidates(lifetimes: &[ResourceLifetime]) -> Vec<TransientAliasCandidate> {
    let mut candidates = Vec::new();
    for (left_index, left) in lifetimes.iter().enumerate() {
        for right in lifetimes.iter().skip(left_index + 1) {
            if left.last_pass() < right.first_pass() || right.last_pass() < left.first_pass() {
                candidates.push(TransientAliasCandidate::new(
                    left.resource(),
                    right.resource(),
                ));
            }
        }
    }

    candidates
}

/// Counts adjacent graphics passes that do not yet require a backend split.
fn count_render_pass_merge_candidates(passes: &[GraphPass]) -> usize {
    passes
        .windows(2)
        .filter(|pair| pair[0].writes().iter().any(is_color_output) && !pair[1].side_effect())
        .count()
}

/// Returns whether one output writes a color attachment.
fn is_color_output(output: &PassOutput) -> bool {
    output.state() == ResourceState::ColorAttachment
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that read/write usage generates the standard render target order.
    #[test]
    fn standard_frame_graph_orders_dependencies() {
        let graph = FrameGraphPlan::standard_frame_with_readback(
            [0.0, 0.1, 0.2, 1.0],
            FrameGraphInitialStates::new(
                ResourceState::Undefined,
                [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
                [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
                ResourceState::Undefined,
                ResourceState::Undefined,
                ResourceState::Undefined,
                ResourceState::Undefined,
            ),
            false,
            true,
            true,
        )
        .expect("graph should compile");
        let names = graph
            .passes()
            .iter()
            .map(GraphPass::name)
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        expected.extend(SHADOW_PASSES);
        expected.extend(TRANSLUCENT_SHADOW_PASSES);
        expected.extend([SCENE_PASS, POST_PASS, PRESENT_PASS]);

        assert_eq!(names, expected);
        assert_eq!(graph.resource_count(), 13);
        assert!(graph.transition_count() >= 10);
        assert!(graph.barrier_count() >= 6);
    }

    // Verifies that persistent resource states affect the next frame's barrier plan.
    #[test]
    fn standard_frame_graph_uses_actual_initial_states() {
        let graph = FrameGraphPlan::standard_frame_with_readback(
            [0.0, 0.1, 0.2, 1.0],
            FrameGraphInitialStates::new(
                ResourceState::Present,
                [ResourceState::ShaderRead; SHADOW_CASCADE_COUNT],
                [ResourceState::ShaderRead; SHADOW_CASCADE_COUNT],
                ResourceState::ShaderRead,
                ResourceState::ShaderRead,
                ResourceState::ShaderRead,
                ResourceState::DepthAttachment,
            ),
            false,
            true,
            true,
        )
        .expect("graph should compile");

        assert!(graph.barriers().iter().any(|barrier| {
            barrier.resource() == GraphResource::SceneColor
                && barrier.from() == ResourceState::ShaderRead
                && barrier.to() == ResourceState::ColorAttachment
                && barrier.location() == BarrierLocation::BeforePass(SCENE_PASS)
        }));
        assert!(!graph.barriers().iter().any(|barrier| {
            barrier.resource() == GraphResource::SceneDepth
                && barrier.location() == BarrierLocation::BeforePass(SCENE_PASS)
        }));
        assert!(graph.barriers().iter().any(|barrier| {
            barrier.resource() == GraphResource::SceneNormalRoughness
                && barrier.from() == ResourceState::ShaderRead
                && barrier.to() == ResourceState::ColorAttachment
                && barrier.location() == BarrierLocation::BeforePass(SCENE_PASS)
        }));
        assert_eq!(
            graph.final_state_for(GraphResource::SwapchainImage),
            Some(ResourceState::Present)
        );
        assert_eq!(
            graph.final_state_for(GraphResource::ShadowCascade0),
            Some(ResourceState::ShaderRead)
        );
        assert_eq!(
            graph.final_state_for(GraphResource::TranslucentShadow0),
            Some(ResourceState::ShaderRead)
        );
    }

    // Verifies that cached shadow maps stay readable while expensive shadow write passes are omitted.
    #[test]
    fn standard_frame_graph_reuses_cached_shadow_maps() {
        let graph = FrameGraphPlan::standard_frame_with_shadow_refresh(
            [0.0, 0.1, 0.2, 1.0],
            FrameGraphInitialStates::new(
                ResourceState::Present,
                [ResourceState::ShaderRead; SHADOW_CASCADE_COUNT],
                [ResourceState::ShaderRead; SHADOW_CASCADE_COUNT],
                ResourceState::ShaderRead,
                ResourceState::ShaderRead,
                ResourceState::ShaderRead,
                ResourceState::ShaderRead,
            ),
            false,
            true,
            true,
            false,
        )
        .expect("graph should compile");
        let names = graph
            .passes()
            .iter()
            .map(GraphPass::name)
            .collect::<Vec<_>>();

        assert_eq!(names, [SCENE_PASS, POST_PASS, PRESENT_PASS]);
        assert_eq!(
            graph.final_state_for(GraphResource::ShadowCascade0),
            Some(ResourceState::ShaderRead)
        );
        assert_eq!(
            graph.final_state_for(GraphResource::TranslucentShadow0),
            Some(ResourceState::ShaderRead)
        );
    }

    // Verifies that translucent shadow passes sample opaque cascade depth explicitly.
    #[test]
    fn standard_frame_graph_declares_translucent_shadow_shader_reads() {
        let graph = FrameGraphPlan::standard_frame_with_readback(
            [0.0, 0.1, 0.2, 1.0],
            FrameGraphInitialStates::new(
                ResourceState::Undefined,
                [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
                [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
                ResourceState::Undefined,
                ResourceState::Undefined,
                ResourceState::Undefined,
                ResourceState::Undefined,
            ),
            false,
            true,
            true,
        )
        .expect("graph should compile");

        assert!(graph.barriers().iter().any(|barrier| {
            barrier.resource() == GraphResource::ShadowCascade0
                && barrier.from() == ResourceState::DepthAttachment
                && barrier.to() == ResourceState::ShaderRead
                && barrier.location() == BarrierLocation::BeforePass(TRANSLUCENT_SHADOW_PASSES[0])
        }));
        assert!(!graph.barriers().iter().any(|barrier| {
            barrier.resource() == GraphResource::ShadowCascade0
                && barrier.location() == BarrierLocation::BeforePass(SCENE_PASS)
        }));
    }

    // Verifies that frames without translucent casters skip translucent shadow graph work.
    #[test]
    fn standard_frame_graph_omits_translucent_shadow_work_when_disabled() {
        let graph = FrameGraphPlan::standard_frame_with_readback(
            [0.0, 0.1, 0.2, 1.0],
            FrameGraphInitialStates::new(
                ResourceState::Undefined,
                [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
                [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
                ResourceState::Undefined,
                ResourceState::Undefined,
                ResourceState::Undefined,
                ResourceState::Undefined,
            ),
            false,
            true,
            false,
        )
        .expect("graph should compile");
        let names = graph
            .passes()
            .iter()
            .map(GraphPass::name)
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        expected.extend(SHADOW_PASSES);
        expected.extend([SCENE_PASS, POST_PASS, PRESENT_PASS]);

        assert_eq!(names, expected);
        assert_eq!(graph.resource_count(), 9);
        assert_eq!(
            graph.final_state_for(GraphResource::TranslucentShadow0),
            None
        );
        assert!(
            !graph
                .barriers()
                .iter()
                .any(|barrier| matches!(barrier.resource(), GraphResource::TranslucentShadow0))
        );
    }

    // Verifies that frames without any shadow casters skip all shadow graph work.
    #[test]
    fn standard_frame_graph_omits_all_shadow_work_without_shadow_casters() {
        let graph = FrameGraphPlan::standard_frame_with_readback(
            [0.0, 0.1, 0.2, 1.0],
            FrameGraphInitialStates::new(
                ResourceState::Undefined,
                [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
                [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
                ResourceState::Undefined,
                ResourceState::Undefined,
                ResourceState::Undefined,
                ResourceState::Undefined,
            ),
            false,
            false,
            false,
        )
        .expect("graph should compile");
        let names = graph
            .passes()
            .iter()
            .map(GraphPass::name)
            .collect::<Vec<_>>();

        assert_eq!(names, [SCENE_PASS, POST_PASS, PRESENT_PASS]);
        assert_eq!(graph.resource_count(), 5);
        assert_eq!(graph.final_state_for(GraphResource::ShadowCascade0), None);
        assert_eq!(
            graph.final_state_for(GraphResource::TranslucentShadow0),
            None
        );
    }

    // Verifies that stale persistent shadow states do not keep shadow resources alive.
    #[test]
    fn standard_frame_graph_ignores_shadow_states_without_shadow_casters() {
        let graph = FrameGraphPlan::standard_frame_with_readback(
            [0.0, 0.1, 0.2, 1.0],
            FrameGraphInitialStates::new(
                ResourceState::Present,
                [ResourceState::ShaderRead; SHADOW_CASCADE_COUNT],
                [ResourceState::ShaderRead; SHADOW_CASCADE_COUNT],
                ResourceState::ShaderRead,
                ResourceState::ShaderRead,
                ResourceState::ShaderRead,
                ResourceState::ShaderRead,
            ),
            false,
            false,
            false,
        )
        .expect("graph should compile");

        assert_eq!(graph.pass_count(), 3);
        assert!(graph.resources().iter().all(|resource| !matches!(
            resource.resource(),
            GraphResource::ShadowCascade0
                | GraphResource::ShadowCascade1
                | GraphResource::ShadowCascade2
                | GraphResource::ShadowCascade3
                | GraphResource::TranslucentShadow0
                | GraphResource::TranslucentShadow1
                | GraphResource::TranslucentShadow2
                | GraphResource::TranslucentShadow3
        )));
    }

    // Verifies that optional framebuffer readback is ordered after post and before present.
    #[test]
    fn standard_frame_graph_places_readback_before_present() {
        let graph = FrameGraphPlan::standard_frame_with_readback(
            [0.0, 0.1, 0.2, 1.0],
            FrameGraphInitialStates::new(
                ResourceState::Undefined,
                [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
                [ResourceState::Undefined; SHADOW_CASCADE_COUNT],
                ResourceState::Undefined,
                ResourceState::Undefined,
                ResourceState::Undefined,
                ResourceState::Undefined,
            ),
            true,
            true,
            true,
        )
        .expect("graph should compile");
        let names = graph
            .passes()
            .iter()
            .map(GraphPass::name)
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        expected.extend(SHADOW_PASSES);
        expected.extend(TRANSLUCENT_SHADOW_PASSES);
        expected.extend([
            SCENE_PASS,
            POST_PASS,
            FRAMEBUFFER_READBACK_PASS,
            PRESENT_PASS,
        ]);

        assert_eq!(names, expected);
        assert!(graph.barriers().iter().any(|barrier| {
            barrier.resource() == GraphResource::SwapchainImage
                && barrier.from() == ResourceState::ColorAttachment
                && barrier.to() == ResourceState::TransferSrc
                && barrier.location() == BarrierLocation::BeforePass(FRAMEBUFFER_READBACK_PASS)
        }));
    }

    // Verifies that unused passes are removed before dependency ordering.
    #[test]
    fn graph_culls_passes_that_do_not_feed_outputs() {
        let graph = FrameGraphBuilder::new()
            .resource(GraphResourceDecl::new(
                GraphResource::SwapchainImage,
                ResourceState::Undefined,
                ResourceState::Present,
            ))
            .resource(GraphResourceDecl::new(
                GraphResource::SceneColor,
                ResourceState::Undefined,
                ResourceState::ShaderRead,
            ))
            .pass(GraphPass::new("unused").write(PassOutput::color_clear(
                GraphResource::SceneColor,
                [1.0, 0.0, 0.0, 1.0],
            )))
            .pass(
                GraphPass::new(PRESENT_PASS)
                    .read(PassInput::read(
                        GraphResource::SwapchainImage,
                        ResourceState::Present,
                    ))
                    .with_side_effect(),
            )
            .compile()
            .expect("graph should compile");
        let names = graph
            .passes()
            .iter()
            .map(GraphPass::name)
            .collect::<Vec<_>>();

        assert_eq!(names, [PRESENT_PASS]);
    }

    // Verifies that pass declarations cannot reference resources absent from the graph.
    #[test]
    fn graph_rejects_missing_resource_usage() {
        let error = FrameGraphBuilder::new()
            .pass(
                GraphPass::new(POST_PASS)
                    .read(PassInput::read(
                        GraphResource::SceneColor,
                        ResourceState::ShaderRead,
                    ))
                    .write(PassOutput::color_load(GraphResource::SwapchainImage)),
            )
            .compile()
            .expect_err("missing resource should fail");

        assert_eq!(
            error,
            GraphCompileError::MissingResource {
                pass: POST_PASS,
                resource: GraphResource::SceneColor.name()
            }
        );
    }
}
