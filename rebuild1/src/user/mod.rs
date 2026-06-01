use std::time::Duration;

use async_trait::async_trait;
use thiserror::Error;

use crate::protocol::{CommandSink, RendererEventEnvelope, TransportError};

#[derive(Clone, Debug)]
pub enum AppEvent {
    Started,
    RedrawRequested,
    ShutdownRequested,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameTime {
    delta: Duration,
}

impl FrameTime {
    /// Creates frame timing data from one measured update delta.
    pub fn new(delta: Duration) -> Self {
        Self { delta }
    }

    /// Returns the measured update delta.
    pub fn delta(self) -> Duration {
        self.delta
    }
}

#[derive(Debug, Error)]
pub enum UserError {
    #[error("failed to send renderer command: {0}")]
    Transport(#[from] TransportError),
}

#[async_trait]
pub trait UserApp: Send {
    /// Sends initial renderer commands required by this user app.
    async fn init(&mut self, out: &CommandSink) -> Result<(), UserError>;

    /// Handles one app event and may emit renderer commands.
    async fn handle_event(&mut self, event: AppEvent, out: &CommandSink) -> Result<(), UserError>;

    /// Advances user state by one frame and may emit renderer commands.
    async fn update(&mut self, time: FrameTime, out: &CommandSink) -> Result<(), UserError>;

    /// Handles one renderer event and may emit follow-up renderer commands.
    async fn handle_renderer_event(
        &mut self,
        event: RendererEventEnvelope,
        out: &CommandSink,
    ) -> Result<(), UserError>;
}

#[derive(Default)]
pub struct NoopUserApp;

#[async_trait]
impl UserApp for NoopUserApp {
    /// Performs no initialization and emits no renderer commands.
    async fn init(&mut self, _out: &CommandSink) -> Result<(), UserError> {
        Ok(())
    }

    /// Ignores one app event and emits no renderer commands.
    async fn handle_event(
        &mut self,
        _event: AppEvent,
        _out: &CommandSink,
    ) -> Result<(), UserError> {
        Ok(())
    }

    /// Ignores one frame update and emits no renderer commands.
    async fn update(&mut self, _time: FrameTime, _out: &CommandSink) -> Result<(), UserError> {
        Ok(())
    }

    /// Ignores one renderer event and emits no follow-up commands.
    async fn handle_renderer_event(
        &mut self,
        _event: RendererEventEnvelope,
        _out: &CommandSink,
    ) -> Result<(), UserError> {
        Ok(())
    }
}
