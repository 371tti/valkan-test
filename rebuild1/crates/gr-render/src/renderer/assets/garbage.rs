use crate::protocol::AssetHandle;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeferredAssetDestroy {
    asset: AssetHandle,
    retire_after_submission: u64,
}

impl DeferredAssetDestroy {
    /// Records one asset handle and the newest queue submission that may still reference it.
    pub(crate) fn new(asset: AssetHandle, retire_after_submission: u64) -> Self {
        Self {
            asset,
            retire_after_submission,
        }
    }

    /// Returns the asset handle whose GPU resources may now be destroyed.
    pub(crate) fn asset(self) -> AssetHandle {
        self.asset
    }

    /// Returns the frame submission that must complete before backend destruction.
    pub(crate) fn retire_after_submission(self) -> u64 {
        self.retire_after_submission
    }
}

#[derive(Default)]
pub(crate) struct DeferredDestroyQueue {
    pending: Vec<DeferredAssetDestroy>,
}

impl DeferredDestroyQueue {
    /// Queues one retired asset until every frame that could reference it has completed.
    pub(crate) fn defer(&mut self, asset: AssetHandle, retire_after_submission: u64) {
        self.pending
            .push(DeferredAssetDestroy::new(asset, retire_after_submission));
    }

    /// Returns only assets whose last possible frame reference is fence-complete.
    pub(crate) fn collect_ready(&mut self, completed_submission: u64) -> Vec<DeferredAssetDestroy> {
        let mut ready = Vec::new();
        let mut pending = Vec::with_capacity(self.pending.len());
        for destroy in self.pending.drain(..) {
            if destroy.retire_after_submission() <= completed_submission {
                ready.push(destroy);
            } else {
                pending.push(destroy);
            }
        }
        self.pending = pending;
        ready
    }

    /// Returns how many retired assets are waiting for destruction.
    pub(crate) fn len(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{MeshHandle, TextureHandle};

    #[test]
    fn deferred_assets_wait_for_their_submission_fence() {
        let first = AssetHandle::Mesh(MeshHandle::from_raw(1).unwrap());
        let second = AssetHandle::Texture(TextureHandle::from_raw(2).unwrap());
        let mut queue = DeferredDestroyQueue::default();
        queue.defer(first, 3);
        queue.defer(second, 5);

        assert!(queue.collect_ready(2).is_empty());
        assert_eq!(queue.len(), 2);
        assert_eq!(
            queue
                .collect_ready(3)
                .into_iter()
                .map(DeferredAssetDestroy::asset)
                .collect::<Vec<_>>(),
            vec![first]
        );
        assert_eq!(queue.len(), 1);
        assert_eq!(
            queue
                .collect_ready(5)
                .into_iter()
                .map(DeferredAssetDestroy::asset)
                .collect::<Vec<_>>(),
            vec![second]
        );
        assert_eq!(queue.len(), 0);
    }
}
