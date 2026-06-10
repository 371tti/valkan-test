use std::num::NonZeroU64;

macro_rules! typed_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(NonZeroU64);

        impl $name {
            /// Creates this id from a non-zero raw value.
            pub fn new(value: NonZeroU64) -> Self {
                Self(value)
            }

            /// Creates this id from a raw integer and rejects zero.
            pub fn from_raw(value: u64) -> Option<Self> {
                NonZeroU64::new(value).map(Self)
            }

            /// Returns the non-zero raw value used by protocol logs.
            pub fn raw(self) -> u64 {
                self.0.get()
            }
        }
    };
}

typed_id!(RequestId);
typed_id!(FrameId);
typed_id!(SceneHandle);
typed_id!(MeshHandle);
typed_id!(MaterialHandle);
typed_id!(TextureHandle);
typed_id!(PipelineHandle);
typed_id!(ExternalObjectId);
typed_id!(SurfaceId);
typed_id!(SurfaceGeneration);
typed_id!(WindowId);
typed_id!(ViewId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProtocolVersion(u16);

impl ProtocolVersion {
    /// Returns the protocol version understood by this crate.
    pub fn current() -> Self {
        Self(1)
    }

    /// Returns the integer version written into command and event logs.
    pub fn raw(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exposure(f32);

impl Exposure {
    /// Creates a finite non-negative exposure value.
    pub fn new(value: f32) -> Option<Self> {
        (value.is_finite() && value >= 0.0).then_some(Self(value))
    }

    /// Returns the finite non-negative exposure value.
    pub fn value(self) -> f32 {
        self.0
    }
}

impl Default for Exposure {
    /// Creates the default neutral exposure value.
    fn default() -> Self {
        Self(1.0)
    }
}
