use std::{future::Future, time::Duration};

use thiserror::Error;

use gr_render::protocol::{CommandSink, RendererEventEnvelope, TransportError};

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

pub trait UserApp: Send {
    /// Sends initial renderer commands required by this user app.
    fn init<'a>(
        &'a mut self,
        out: &'a CommandSink,
    ) -> impl Future<Output = Result<(), UserError>> + Send + 'a;

    /// Handles one app event and may emit renderer commands.
    fn handle_event<'a>(
        &'a mut self,
        event: AppEvent,
        out: &'a CommandSink,
    ) -> impl Future<Output = Result<(), UserError>> + Send + 'a;

    /// Advances user state by one frame and may emit renderer commands.
    fn update<'a>(
        &'a mut self,
        time: FrameTime,
        out: &'a CommandSink,
    ) -> impl Future<Output = Result<(), UserError>> + Send + 'a;

    /// Handles one renderer event and may emit follow-up renderer commands.
    fn handle_renderer_event<'a>(
        &'a mut self,
        event: RendererEventEnvelope,
        out: &'a CommandSink,
    ) -> impl Future<Output = Result<(), UserError>> + Send + 'a;
}

#[derive(Default)]
pub struct NoopUserApp;

impl UserApp for NoopUserApp {
    /// Performs no initialization and emits no renderer commands.
    async fn init<'a>(&'a mut self, _out: &'a CommandSink) -> Result<(), UserError> {
        Ok(())
    }

    /// Ignores one app event and emits no renderer commands.
    async fn handle_event<'a>(
        &'a mut self,
        _event: AppEvent,
        _out: &'a CommandSink,
    ) -> Result<(), UserError> {
        Ok(())
    }

    /// Ignores one frame update and emits no renderer commands.
    async fn update<'a>(
        &'a mut self,
        _time: FrameTime,
        _out: &'a CommandSink,
    ) -> Result<(), UserError> {
        Ok(())
    }

    /// Ignores one renderer event and emits no follow-up commands.
    async fn handle_renderer_event<'a>(
        &'a mut self,
        _event: RendererEventEnvelope,
        _out: &'a CommandSink,
    ) -> Result<(), UserError> {
        Ok(())
    }
}
