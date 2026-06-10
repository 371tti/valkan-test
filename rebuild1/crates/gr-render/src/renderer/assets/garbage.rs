use crate::protocol::AssetHandle;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeferredAssetDestroy {
    asset: AssetHandle,
}

impl DeferredAssetDestroy {
    /// Records one asset handle that must wait for GPU lifetime retirement.
    pub(crate) fn new(asset: AssetHandle) -> Self {
        Self { asset }
    }

    /// Returns the asset handle whose GPU resources may now be destroyed.
    pub(crate) fn asset(self) -> AssetHandle {
        self.asset
    }
}

#[derive(Default)]
pub(crate) struct DeferredDestroyQueue {
    pending: Vec<DeferredAssetDestroy>,
}

impl DeferredDestroyQueue {
    /// Queues one retired asset handle instead of destroying GPU resources inline.
    pub(crate) fn defer(&mut self, asset: AssetHandle) {
        self.pending.push(DeferredAssetDestroy::new(asset));
    }

    /// Returns currently safe-to-destroy asset handles for the Stage 5 no-GPU skeleton.
    pub(crate) fn collect_ready(&mut self) -> Vec<DeferredAssetDestroy> {
        std::mem::take(&mut self.pending)
    }

    /// Returns how many retired assets are waiting for destruction.
    pub(crate) fn len(&self) -> usize {
        self.pending.len()
    }
}
