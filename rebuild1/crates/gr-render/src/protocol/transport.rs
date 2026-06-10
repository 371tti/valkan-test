use thiserror::Error;
use tokio::sync::mpsc;

use super::{RendererCommandEnvelope, RendererEventEnvelope};

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("transport channel is closed")]
    Closed,
    #[error("transport channel is full")]
    Full,
}

pub struct RendererEndpoint {
    commands: mpsc::Sender<RendererCommandEnvelope>,
    events: mpsc::Receiver<RendererEventEnvelope>,
}

impl RendererEndpoint {
    /// Sends one command to the renderer task and waits for channel capacity.
    pub async fn send(&self, command: RendererCommandEnvelope) -> Result<(), TransportError> {
        tracing::trace!(
            command = command.payload.name(),
            request_id = command.request_id.map(|id| id.raw()),
            frame_id = command.frame_id.map(|id| id.raw()),
            "sending renderer command"
        );

        self.commands
            .send(command)
            .await
            .map_err(|_| TransportError::Closed)
    }

    /// Sends one command from a synchronous callback without waiting.
    pub fn try_send(&self, command: RendererCommandEnvelope) -> Result<(), TransportError> {
        tracing::trace!(
            command = command.payload.name(),
            request_id = command.request_id.map(|id| id.raw()),
            frame_id = command.frame_id.map(|id| id.raw()),
            "try-sending renderer command"
        );

        self.commands
            .try_send(command)
            .map_err(|error| match error {
                mpsc::error::TrySendError::Closed(_) => TransportError::Closed,
                mpsc::error::TrySendError::Full(_) => TransportError::Full,
            })
    }

    /// Receives the next renderer event from the renderer task.
    pub async fn recv_event(&mut self) -> Option<RendererEventEnvelope> {
        let event = self.events.recv().await;
        if let Some(event) = &event {
            tracing::trace!(
                event = event.payload.name(),
                request_id = event.request_id.map(|id| id.raw()),
                frame_id = event.frame_id.map(|id| id.raw()),
                "received renderer event"
            );
        }

        event
    }

    /// Receives one renderer event without waiting for async context.
    pub fn try_recv_event(&mut self) -> Option<RendererEventEnvelope> {
        let event = self.events.try_recv().ok();
        if let Some(event) = &event {
            tracing::trace!(
                event = event.payload.name(),
                request_id = event.request_id.map(|id| id.raw()),
                frame_id = event.frame_id.map(|id| id.raw()),
                "try-received renderer event"
            );
        }

        event
    }

    /// Creates a command sink that can be moved into user code.
    pub fn command_sink(&self) -> CommandSink {
        CommandSink {
            commands: self.commands.clone(),
        }
    }
}

#[derive(Clone)]
pub struct CommandSink {
    commands: mpsc::Sender<RendererCommandEnvelope>,
}

impl CommandSink {
    /// Sends one command through this cloned command-only sink.
    pub async fn send(&self, command: RendererCommandEnvelope) -> Result<(), TransportError> {
        tracing::trace!(
            command = command.payload.name(),
            request_id = command.request_id.map(|id| id.raw()),
            frame_id = command.frame_id.map(|id| id.raw()),
            "sending renderer command through sink"
        );

        self.commands
            .send(command)
            .await
            .map_err(|_| TransportError::Closed)
    }

    /// Sends one command without waiting for channel capacity.
    pub fn try_send(&self, command: RendererCommandEnvelope) -> Result<(), TransportError> {
        tracing::trace!(
            command = command.payload.name(),
            request_id = command.request_id.map(|id| id.raw()),
            frame_id = command.frame_id.map(|id| id.raw()),
            "try-sending renderer command through sink"
        );

        self.commands
            .try_send(command)
            .map_err(|error| match error {
                mpsc::error::TrySendError::Closed(_) => TransportError::Closed,
                mpsc::error::TrySendError::Full(_) => TransportError::Full,
            })
    }
}

pub struct RendererInbox {
    commands: mpsc::Receiver<RendererCommandEnvelope>,
    events: mpsc::Sender<RendererEventEnvelope>,
}

impl RendererInbox {
    /// Receives the next command sent to the renderer task.
    pub async fn recv_command(&mut self) -> Option<RendererCommandEnvelope> {
        let command = self.commands.recv().await;
        if let Some(command) = &command {
            tracing::trace!(
                command = command.payload.name(),
                request_id = command.request_id.map(|id| id.raw()),
                frame_id = command.frame_id.map(|id| id.raw()),
                "renderer received command"
            );
        }

        command
    }

    /// Sends one event back to user code and tools.
    pub async fn send_event(&self, event: RendererEventEnvelope) -> Result<(), TransportError> {
        tracing::trace!(
            event = event.payload.name(),
            request_id = event.request_id.map(|id| id.raw()),
            frame_id = event.frame_id.map(|id| id.raw()),
            "renderer sending event"
        );

        self.events
            .send(event)
            .await
            .map_err(|_| TransportError::Closed)
    }
}

/// Creates the bounded channels that connect user code and renderer task.
pub fn renderer_transport(capacity: usize) -> (RendererEndpoint, RendererInbox) {
    let capacity = capacity.max(1);
    tracing::trace!(capacity, "creating renderer transport");

    let (commands_tx, commands_rx) = mpsc::channel(capacity);
    let (events_tx, events_rx) = mpsc::channel(capacity);

    (
        RendererEndpoint {
            commands: commands_tx,
            events: events_rx,
        },
        RendererInbox {
            commands: commands_rx,
            events: events_tx,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{MessageEnvelope, RendererCommand};

    // Verifies that capacity zero is constrained to a usable bounded channel.
    #[tokio::test]
    async fn zero_capacity_creates_usable_transport() {
        let (endpoint, mut inbox) = renderer_transport(0);

        endpoint
            .send(MessageEnvelope::new(RendererCommand::Shutdown))
            .await
            .expect("transport should accept one command");

        assert!(inbox.recv_command().await.is_some());
    }
}
