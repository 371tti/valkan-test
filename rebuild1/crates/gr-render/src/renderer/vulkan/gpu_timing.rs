use ash::{Device, vk};

/// One start query, up to 62 pass checkpoints, and one final command-buffer query.
///
/// The largest standard graph currently needs 33 checkpoints. Keeping spare capacity makes the
/// optional profiler robust to new passes without making its query pool meaningfully expensive.
const QUERIES_PER_FRAME: u32 = 64;
const RESERVED_FRAME_END_QUERIES: u32 = 1;
const MAX_PASS_CHECKPOINTS: u32 = QUERIES_PER_FRAME - 1 - RESERVED_FRAME_END_QUERIES;

#[derive(Default)]
struct FrameQueryState {
    pass_labels: Vec<&'static str>,
    written_queries: u32,
    dropped_checkpoints: u32,
    complete: bool,
}

impl FrameQueryState {
    fn with_capacity() -> Self {
        Self {
            pass_labels: Vec::with_capacity(MAX_PASS_CHECKPOINTS as usize),
            ..Self::default()
        }
    }

    fn begin(&mut self) {
        self.pass_labels.clear();
        self.written_queries = 1;
        self.dropped_checkpoints = 0;
        self.complete = false;
    }

    /// Reserves the next query while always leaving one query for the command-buffer end.
    fn reserve_checkpoint(&mut self, label: &'static str) -> Option<u32> {
        if self.written_queries >= QUERIES_PER_FRAME - RESERVED_FRAME_END_QUERIES {
            self.dropped_checkpoints = self.dropped_checkpoints.saturating_add(1);
            return None;
        }

        let query = self.written_queries;
        self.written_queries += 1;
        self.pass_labels.push(label);
        Some(query)
    }

    fn reserve_end(&mut self) -> Option<u32> {
        if self.written_queries >= QUERIES_PER_FRAME {
            return None;
        }

        let query = self.written_queries;
        self.written_queries += 1;
        self.complete = true;
        Some(query)
    }
}

/// Optional GPU frame timer used for detailed pass tracing.
///
/// CPU submit, fence, and present durations are intentionally not used here: they mix rendering
/// cost with queue depth, swapchain acquisition, and display pacing. Vulkan timestamps bracket the
/// recorded command buffer and every graph pass. Results and their labels are owned per frame slot
/// and read only after that slot's fence has completed.
pub(super) struct GpuFrameTimer {
    query_pool: vk::QueryPool,
    timestamp_period_ns: f32,
    timestamp_valid_bits: u32,
    slots: Vec<FrameQueryState>,
}

impl GpuFrameTimer {
    /// Creates timestamp queries only when detailed GPU tracing is enabled.
    pub(super) fn create_if_enabled(
        device: &Device,
        frame_slot_count: usize,
        timestamp_period_ns: f32,
        timestamp_valid_bits: u32,
    ) -> Option<Self> {
        if !tracing::enabled!(
            target: "gr_render::renderer::vulkan::gpu_timing",
            tracing::Level::TRACE
        ) {
            return None;
        }

        if timestamp_valid_bits == 0
            || !timestamp_period_ns.is_finite()
            || timestamp_period_ns <= 0.0
        {
            tracing::trace!(
                target: "gr_render::renderer::vulkan::gpu_timing",
                timestamp_valid_bits,
                timestamp_period_ns,
                "GPU timestamps are unavailable for the selected graphics queue"
            );
            return None;
        }

        let query_count = u32::try_from(frame_slot_count)
            .ok()
            .and_then(|count| count.checked_mul(QUERIES_PER_FRAME))?;
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(query_count);
        // Safety: `create_info` contains no borrowed arrays and no custom allocator is used.
        let query_pool = match unsafe { device.create_query_pool(&create_info, None) } {
            Ok(query_pool) => query_pool,
            Err(error) => {
                tracing::warn!(
                    target: "gr_render::renderer::vulkan::gpu_timing",
                    error = ?error,
                    "failed to create optional GPU timestamp queries"
                );
                return None;
            }
        };

        tracing::trace!(
            target: "gr_render::renderer::vulkan::gpu_timing",
            frame_slot_count,
            queries_per_frame = QUERIES_PER_FRAME,
            max_pass_checkpoints = MAX_PASS_CHECKPOINTS,
            timestamp_valid_bits,
            timestamp_period_ns,
            "enabled Vulkan GPU frame timing"
        );
        Some(Self {
            query_pool,
            timestamp_period_ns,
            timestamp_valid_bits,
            slots: (0..frame_slot_count)
                .map(|_| FrameQueryState::with_capacity())
                .collect(),
        })
    }

    /// Resets this slot's query range and records the beginning of its GPU command buffer.
    pub(super) fn record_start(
        &mut self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        frame_slot: usize,
    ) {
        let first_query = first_query_for_slot(frame_slot);
        self.slot_mut(frame_slot).begin();
        // Safety: the command buffer is recording, this slot's prior fence completed before reuse,
        // and the pool contains `QUERIES_PER_FRAME` queries for every valid frame slot.
        unsafe {
            device.cmd_reset_query_pool(
                command_buffer,
                self.query_pool,
                first_query,
                QUERIES_PER_FRAME,
            );
            device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                first_query,
            );
        }
    }

    /// Records completion of one synthetic or graph-owned GPU pass.
    ///
    /// The interval includes barriers recorded immediately before the pass. If a future graph grows
    /// beyond the fixed query budget, extra checkpoints are folded into the unprofiled tail while
    /// the final whole-command-buffer measurement remains valid.
    pub(super) fn record_checkpoint(
        &mut self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        frame_slot: usize,
        pass_label: &'static str,
    ) {
        let reservation = self.slot_mut(frame_slot).reserve_checkpoint(pass_label);
        let Some(relative_query) = reservation else {
            let dropped_checkpoints = self.slot(frame_slot).dropped_checkpoints;
            if dropped_checkpoints == 1 {
                tracing::warn!(
                    target: "gr_render::renderer::vulkan::gpu_timing",
                    frame_slot,
                    max_pass_checkpoints = MAX_PASS_CHECKPOINTS,
                    "GPU pass timestamp capacity exceeded; remaining passes will be folded into the tail"
                );
            }
            return;
        };

        // Safety: the command buffer is recording and `reserve_checkpoint` bounds the relative
        // query while reserving the final query for `record_end`.
        unsafe {
            device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                first_query_for_slot(frame_slot) + relative_query,
            );
        }
    }

    /// Records the point at which every command in this frame has finished on the GPU.
    pub(super) fn record_end(
        &mut self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        frame_slot: usize,
    ) {
        let Some(relative_query) = self.slot_mut(frame_slot).reserve_end() else {
            tracing::warn!(
                target: "gr_render::renderer::vulkan::gpu_timing",
                frame_slot,
                "GPU timestamp end query was unexpectedly exhausted"
            );
            return;
        };

        // Safety: the command buffer is recording and the relative query remains inside this
        // slot's fixed range.
        unsafe {
            device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                first_query_for_slot(frame_slot) + relative_query,
            );
        }
    }

    /// Emits one completed frame and its pass measurements after the slot fence signals.
    pub(super) fn trace_completed(
        &mut self,
        device: &Device,
        frame_slot: usize,
        frame_id: Option<u64>,
    ) {
        let state = self.slot(frame_slot);
        if !state.complete || state.written_queries < 2 {
            tracing::trace!(
                target: "gr_render::renderer::vulkan::gpu_timing",
                frame_id,
                frame_slot,
                "GPU timestamp recording was incomplete for submitted frame"
            );
            return;
        }

        let mut timestamps = vec![0_u64; state.written_queries as usize];
        // Safety: the caller waits for this frame slot's submission fence before reading, the
        // query range belongs to the slot, and TYPE_64 matches the u64 result storage.
        match unsafe {
            device.get_query_pool_results(
                self.query_pool,
                first_query_for_slot(frame_slot),
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64,
            )
        } {
            Ok(()) => self.trace_timestamps(frame_slot, frame_id, &timestamps),
            Err(vk::Result::NOT_READY) => tracing::trace!(
                target: "gr_render::renderer::vulkan::gpu_timing",
                frame_id,
                frame_slot,
                "GPU timestamps were unexpectedly not ready after fence completion"
            ),
            Err(error) => tracing::warn!(
                target: "gr_render::renderer::vulkan::gpu_timing",
                frame_id,
                frame_slot,
                error = ?error,
                "failed to read optional GPU timestamps"
            ),
        }
    }

    fn trace_timestamps(&mut self, frame_slot: usize, frame_id: Option<u64>, timestamps: &[u64]) {
        let timestamp_ticks = elapsed_timestamp_ticks(
            timestamps[0],
            timestamps[timestamps.len() - 1],
            self.timestamp_valid_bits,
        );
        let gpu_frame_ms = self.timestamp_ticks_to_ms(timestamp_ticks);
        let state = self.slot(frame_slot);
        tracing::trace!(
            target: "gr_render::renderer::vulkan::gpu_timing",
            frame_id,
            frame_slot,
            gpu_frame_ms,
            timestamp_ticks,
            pass_checkpoint_count = state.pass_labels.len(),
            dropped_checkpoint_count = state.dropped_checkpoints,
            wsi_present_included = false,
            "measured GPU command-buffer frame time"
        );

        for (pass_index, &pass_label) in state.pass_labels.iter().enumerate() {
            let start = timestamps[pass_index];
            let end = timestamps[pass_index + 1];
            let pass_ticks = elapsed_timestamp_ticks(start, end, self.timestamp_valid_bits);
            let gpu_pass_ms = self.timestamp_ticks_to_ms(pass_ticks);
            tracing::trace!(
                target: "gr_render::renderer::vulkan::gpu_timing",
                frame_id,
                frame_slot,
                pass_index,
                gpu_pass = pass_label,
                gpu_pass_ms,
                pass_ticks,
                "measured GPU pass time"
            );
        }

        let tail_start = state.pass_labels.len();
        let tail_ticks = elapsed_timestamp_ticks(
            timestamps[tail_start],
            timestamps[timestamps.len() - 1],
            self.timestamp_valid_bits,
        );
        let gpu_command_buffer_tail_ms = self.timestamp_ticks_to_ms(tail_ticks);
        tracing::trace!(
            target: "gr_render::renderer::vulkan::gpu_timing",
            frame_id,
            frame_slot,
            gpu_command_buffer_tail_ms,
            tail_ticks,
            dropped_checkpoint_count = state.dropped_checkpoints,
            "measured GPU work after the final recorded pass"
        );
    }

    fn timestamp_ticks_to_ms(&self, ticks: u64) -> f64 {
        ticks as f64 * f64::from(self.timestamp_period_ns) / 1_000_000.0
    }

    fn slot(&self, frame_slot: usize) -> &FrameQueryState {
        self.slots
            .get(frame_slot)
            .expect("frame slot count matches GPU timestamp slot count")
    }

    fn slot_mut(&mut self, frame_slot: usize) -> &mut FrameQueryState {
        self.slots
            .get_mut(frame_slot)
            .expect("frame slot count matches GPU timestamp slot count")
    }

    /// Destroys the timestamp query pool after all frame fences have completed.
    pub(super) fn destroy(self, device: &Device) {
        // Safety: the owning frame system waits for submitted work before teardown, and no custom
        // allocator was used at creation.
        unsafe {
            device.destroy_query_pool(self.query_pool, None);
        }
    }
}

fn first_query_for_slot(frame_slot: usize) -> u32 {
    u32::try_from(frame_slot).expect("frame slot count fits Vulkan query indexing")
        * QUERIES_PER_FRAME
}

pub(super) fn elapsed_timestamp_ticks(start: u64, end: u64, valid_bits: u32) -> u64 {
    let elapsed = end.wrapping_sub(start);
    if valid_bits >= u64::BITS {
        elapsed
    } else {
        elapsed & ((1_u64 << valid_bits) - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FrameQueryState, MAX_PASS_CHECKPOINTS, QUERIES_PER_FRAME, elapsed_timestamp_ticks,
    };

    #[test]
    fn elapsed_timestamp_ticks_handles_full_width_counter() {
        assert_eq!(elapsed_timestamp_ticks(120, 345, 64), 225);
    }

    #[test]
    fn elapsed_timestamp_ticks_handles_valid_bit_wraparound() {
        assert_eq!(elapsed_timestamp_ticks(250, 5, 8), 11);
    }

    #[test]
    fn query_capacity_always_reserves_the_frame_end() {
        let mut state = FrameQueryState::with_capacity();
        state.begin();
        for expected_query in 1..=MAX_PASS_CHECKPOINTS {
            assert_eq!(state.reserve_checkpoint("pass"), Some(expected_query));
        }
        assert_eq!(state.reserve_checkpoint("overflow"), None);
        assert_eq!(state.dropped_checkpoints, 1);
        assert_eq!(state.reserve_end(), Some(QUERIES_PER_FRAME - 1));
        assert_eq!(state.written_queries, QUERIES_PER_FRAME);
        assert!(state.complete);
    }

    #[test]
    fn labels_are_owned_independently_per_frame_slot() {
        let mut first = FrameQueryState::with_capacity();
        let mut second = FrameQueryState::with_capacity();
        first.begin();
        second.begin();
        first.reserve_checkpoint("scene");
        second.reserve_checkpoint("post");

        assert_eq!(first.pass_labels, ["scene"]);
        assert_eq!(second.pass_labels, ["post"]);
        first.begin();
        assert!(first.pass_labels.is_empty());
        assert_eq!(second.pass_labels, ["post"]);
    }
}
