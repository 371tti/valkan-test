use super::{
    PLANAR_REFLECTION_UPDATE_INTERVAL, REFLECTION_PROBE_UPDATE_INTERVAL, SHADOW_UPDATE_INTERVAL,
    shadows,
};

pub(super) struct PassSchedule {
    cached_shadow: Option<shadows::PreparedShadow>,
    shadow_frames_until_update: u32,
    reflection_probe_frames_until_update: u32,
    planar_reflection_frames_until_update: u32,
}

#[derive(Clone, Copy)]
pub(super) struct PassUpdates {
    pub shadow_map: bool,
    pub reflection_probe: bool,
    pub planar_reflection: bool,
}

impl PassSchedule {
    pub(super) fn new() -> Self {
        Self {
            cached_shadow: None,
            shadow_frames_until_update: 0,
            reflection_probe_frames_until_update: 0,
            planar_reflection_frames_until_update: 0,
        }
    }

    pub(super) fn needs_shadow_update(&self) -> bool {
        self.cached_shadow.is_none() || self.shadow_frames_until_update == 0
    }

    pub(super) fn shadow_frame(
        &mut self,
        prepared: Option<shadows::PreparedShadow>,
    ) -> (shadows::PreparedShadow, bool) {
        let should_update = self.needs_shadow_update();

        if should_update {
            let prepared = prepared.expect("shadow update requested without prepared shadow data");
            self.cached_shadow = Some(prepared);
            self.shadow_frames_until_update = SHADOW_UPDATE_INTERVAL.saturating_sub(1);
            (prepared, true)
        } else {
            self.shadow_frames_until_update = self.shadow_frames_until_update.saturating_sub(1);
            (
                self.cached_shadow
                    .expect("shadow cache missing while update was deferred"),
                false,
            )
        }
    }

    pub(super) fn reset_scene_dependent(&mut self) {
        self.cached_shadow = None;
        self.shadow_frames_until_update = 0;
        self.reflection_probe_frames_until_update = 0;
        self.planar_reflection_frames_until_update = 0;
    }

    pub(super) fn reset_reflection_probe(&mut self) {
        self.reflection_probe_frames_until_update = 0;
    }

    pub(super) fn reset_planar_reflection(&mut self) {
        self.planar_reflection_frames_until_update = 0;
    }

    pub(super) fn should_update_reflection_probe(&mut self, enabled: bool) -> bool {
        update_due(
            &mut self.reflection_probe_frames_until_update,
            enabled,
            REFLECTION_PROBE_UPDATE_INTERVAL,
        )
    }

    pub(super) fn should_update_planar_reflection(&mut self, enabled: bool) -> bool {
        update_due(
            &mut self.planar_reflection_frames_until_update,
            enabled,
            PLANAR_REFLECTION_UPDATE_INTERVAL,
        )
    }
}

fn update_due(counter: &mut u32, enabled: bool, interval: u32) -> bool {
    if !enabled {
        *counter = 0;
        return false;
    }

    if *counter == 0 {
        *counter = interval.saturating_sub(1);
        true
    } else {
        *counter -= 1;
        false
    }
}
