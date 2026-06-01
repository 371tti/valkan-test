use std::collections::BTreeMap;

use crate::protocol::{
    DropReason, NativeSurfaceHandle, NonZeroExtent, SurfaceDescriptor, SurfaceGeneration,
    SurfaceId, WindowId,
};

#[derive(Default)]
pub(crate) struct SurfaceRegistry {
    surfaces: BTreeMap<SurfaceId, SurfaceState>,
}

impl SurfaceRegistry {
    /// Registers or replaces the native surface state for one protocol window.
    pub(crate) fn configure(&mut self, descriptor: SurfaceDescriptor) -> SurfaceState {
        let state = SurfaceState::from_descriptor(descriptor);
        tracing::trace!(
            window_id = state.window_id.raw(),
            surface_id = state.surface_id.raw(),
            generation = state.generation.raw(),
            width = state.extent.width(),
            height = state.extent.height(),
            platform = state.native.platform().name(),
            "registered renderer surface"
        );
        self.surfaces.insert(state.surface_id, state);
        state
    }

    /// Updates the drawable extent for a configured surface and reports whether it existed.
    pub(crate) fn resize(
        &mut self,
        surface_id: SurfaceId,
        generation: SurfaceGeneration,
        extent: NonZeroExtent,
    ) -> bool {
        let Some(surface) = self.surfaces.get_mut(&surface_id) else {
            tracing::trace!(
                surface_id = surface_id.raw(),
                "renderer surface resize missed"
            );
            return false;
        };

        surface.resize(generation, extent);
        tracing::trace!(
            surface_id = surface_id.raw(),
            generation = generation.raw(),
            width = extent.width(),
            height = extent.height(),
            "updated renderer surface extent"
        );
        true
    }

    /// Returns why a frame should be dropped for its target surface, if it should be dropped.
    pub(crate) fn frame_drop_reason(
        &self,
        surface_id: SurfaceId,
        generation: SurfaceGeneration,
    ) -> Option<DropReason> {
        let Some(surface) = self.surfaces.get(&surface_id) else {
            return Some(DropReason::NoSurface { surface_id });
        };

        (surface.generation != generation).then_some(DropReason::StaleSurfaceGeneration {
            surface_id,
            submitted: generation,
            current: surface.generation,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SurfaceState {
    window_id: WindowId,
    surface_id: SurfaceId,
    generation: SurfaceGeneration,
    extent: NonZeroExtent,
    native: NativeSurfaceHandle,
}

impl SurfaceState {
    /// Creates renderer-owned surface state from one configure command payload.
    fn from_descriptor(descriptor: SurfaceDescriptor) -> Self {
        Self {
            window_id: descriptor.window_id,
            surface_id: descriptor.surface_id,
            generation: descriptor.generation,
            extent: descriptor.extent,
            native: descriptor.native,
        }
    }

    /// Returns the protocol id of this renderer-facing surface.
    pub(crate) fn surface_id(self) -> SurfaceId {
        self.surface_id
    }

    /// Returns the current surface generation accepted by the renderer.
    pub(crate) fn generation(self) -> SurfaceGeneration {
        self.generation
    }

    /// Returns the current validated drawable extent.
    pub(crate) fn extent(self) -> NonZeroExtent {
        self.extent
    }

    /// Returns the sendable native handle captured for surface creation.
    pub(crate) fn native(self) -> NativeSurfaceHandle {
        self.native
    }

    /// Replaces the current drawable extent with a newly validated extent.
    fn resize(&mut self, generation: SurfaceGeneration, extent: NonZeroExtent) {
        self.generation = generation;
        self.extent = extent;
    }
}
