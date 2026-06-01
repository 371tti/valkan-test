use std::num::NonZeroIsize;

use super::{SurfaceGeneration, SurfaceId, WindowId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NonZeroExtent {
    width: u32,
    height: u32,
}

impl NonZeroExtent {
    /// Creates a validated extent and rejects zero-sized surfaces.
    pub fn new(width: u32, height: u32) -> Option<Self> {
        (width > 0 && height > 0).then_some(Self { width, height })
    }

    /// Returns the validated width in pixels.
    pub fn width(self) -> u32 {
        self.width
    }

    /// Returns the validated height in pixels.
    pub fn height(self) -> u32 {
        self.height
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeSurfaceHandle {
    Win32(Win32SurfaceHandle),
}

impl NativeSurfaceHandle {
    /// Returns the platform family represented by this native handle.
    pub fn platform(self) -> NativeSurfacePlatform {
        match self {
            Self::Win32(_) => NativeSurfacePlatform::Win32,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeSurfacePlatform {
    Win32,
}

impl NativeSurfacePlatform {
    /// Returns the compact platform name used by renderer diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::Win32 => "win32",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Win32SurfaceHandle {
    hwnd: NonZeroIsize,
    hinstance: Option<NonZeroIsize>,
}

impl Win32SurfaceHandle {
    /// Creates a sendable copy of the Win32 handles needed for Vulkan surface creation.
    pub fn new(hwnd: NonZeroIsize, hinstance: Option<NonZeroIsize>) -> Self {
        Self { hwnd, hinstance }
    }

    /// Returns the non-zero HWND captured while the window is alive.
    pub fn hwnd(self) -> NonZeroIsize {
        self.hwnd
    }

    /// Returns the optional Win32 instance handle associated with the HWND.
    pub fn hinstance(self) -> Option<NonZeroIsize> {
        self.hinstance
    }
}

#[derive(Clone, Debug)]
pub struct SurfaceDescriptor {
    pub window_id: WindowId,
    pub surface_id: SurfaceId,
    pub generation: SurfaceGeneration,
    pub extent: NonZeroExtent,
    pub native: NativeSurfaceHandle,
}

impl SurfaceDescriptor {
    /// Creates a renderer-facing surface descriptor from validated extent and native handles.
    pub fn new(
        window_id: WindowId,
        surface_id: SurfaceId,
        generation: SurfaceGeneration,
        extent: NonZeroExtent,
        native: NativeSurfaceHandle,
    ) -> Self {
        Self {
            window_id,
            surface_id,
            generation,
            extent,
            native,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that zero-sized extents cannot enter renderer-facing protocol data.
    #[test]
    fn extent_rejects_zero_size() {
        assert!(NonZeroExtent::new(1, 1).is_some());
        assert!(NonZeroExtent::new(0, 1).is_none());
        assert!(NonZeroExtent::new(1, 0).is_none());
    }
}
