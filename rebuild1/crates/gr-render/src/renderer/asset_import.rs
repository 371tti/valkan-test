use std::{collections::VecDeque, path::PathBuf};

use tokio::task::JoinHandle;

use crate::{
    import::{ImportTaskError, ImportedScene, import_asset_on_worker},
    protocol::{RendererCommandEnvelope, RendererInbox, RequestId},
};

struct AssetImportRequest {
    request_id: Option<RequestId>,
    path: PathBuf,
}

struct ActiveAssetImport {
    request_id: Option<RequestId>,
    path: PathBuf,
    task: JoinHandle<Result<ImportedScene, ImportTaskError>>,
}

/// A completed CPU import that is ready for backend-local registration and GPU upload.
pub(super) struct AssetImportCompletion {
    pub(super) request_id: Option<RequestId>,
    pub(super) result: Result<ImportedScene, String>,
}

/// The next input handled by both renderer backends.
pub(super) enum RendererLoopEvent {
    Command(RendererCommandEnvelope),
    AssetImported(AssetImportCompletion),
    CommandChannelClosed,
}

/// Runs at most one CPU import while renderer commands remain responsive.
#[derive(Default)]
pub(super) struct AssetImportScheduler {
    pending: VecDeque<AssetImportRequest>,
    active: Option<ActiveAssetImport>,
}

impl AssetImportScheduler {
    /// Adds one request to the FIFO without starting additional concurrent workers.
    pub(super) fn enqueue(&mut self, request_id: Option<RequestId>, path: PathBuf) {
        tracing::trace!(
            request_id = request_id.map(|id| id.raw()),
            path = %path.display(),
            pending = self.pending.len() + 1,
            active = self.active.is_some(),
            "queued renderer asset import"
        );
        self.pending
            .push_back(AssetImportRequest { request_id, path });
    }

    /// Waits for either a renderer command or the single active CPU import.
    pub(super) async fn next_event(&mut self, inbox: &mut RendererInbox) -> RendererLoopEvent {
        self.start_next_if_idle();

        enum Selected {
            Command(Option<RendererCommandEnvelope>),
            Import(Result<Result<ImportedScene, ImportTaskError>, tokio::task::JoinError>),
        }

        let selected = if let Some(active) = self.active.as_mut() {
            tokio::select! {
                command = inbox.recv_command() => Selected::Command(command),
                result = &mut active.task => Selected::Import(result),
            }
        } else {
            Selected::Command(inbox.recv_command().await)
        };

        match selected {
            Selected::Command(Some(command)) => RendererLoopEvent::Command(command),
            Selected::Command(None) => RendererLoopEvent::CommandChannelClosed,
            Selected::Import(joined) => {
                let active = self
                    .active
                    .take()
                    .expect("an import result requires one active scheduler task");
                let result = match joined {
                    Ok(result) => result.map_err(|error| error.to_string()),
                    Err(error) => Err(format!("asset import scheduler task failed: {error}")),
                };
                tracing::trace!(
                    request_id = active.request_id.map(|id| id.raw()),
                    path = %active.path.display(),
                    succeeded = result.is_ok(),
                    pending = self.pending.len(),
                    "completed renderer asset import"
                );
                RendererLoopEvent::AssetImported(AssetImportCompletion {
                    request_id: active.request_id,
                    result,
                })
            }
        }
    }

    /// Cancels future delivery and prevents any completed CPU data from reaching GPU code.
    pub(super) fn shutdown(&mut self) {
        self.pending.clear();
        if let Some(active) = self.active.take() {
            active.task.abort();
        }
    }

    fn start_next_if_idle(&mut self) {
        if self.active.is_some() {
            return;
        }
        let Some(request) = self.pending.pop_front() else {
            return;
        };

        tracing::trace!(
            request_id = request.request_id.map(|id| id.raw()),
            path = %request.path.display(),
            pending = self.pending.len(),
            "starting renderer asset import"
        );
        let task = tokio::spawn(import_asset_on_worker(request.path.clone()));
        self.active = Some(ActiveAssetImport {
            request_id: request.request_id,
            path: request.path,
            task,
        });
    }
}

impl Drop for AssetImportScheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{MessageEnvelope, RendererCommand, renderer_transport};

    #[tokio::test]
    async fn renderer_commands_remain_responsive_while_one_import_is_active() {
        let (endpoint, mut inbox) = renderer_transport(4);
        let active_task =
            tokio::spawn(std::future::pending::<Result<ImportedScene, ImportTaskError>>());
        let mut scheduler = AssetImportScheduler {
            pending: VecDeque::new(),
            active: Some(ActiveAssetImport {
                request_id: None,
                path: PathBuf::from("pending-test-import"),
                task: active_task,
            }),
        };
        scheduler.enqueue(None, PathBuf::from("queued-test-import"));
        endpoint
            .send(MessageEnvelope::new(RendererCommand::CreateScene))
            .await
            .expect("test command should be queued");

        let RendererLoopEvent::Command(command) = scheduler.next_event(&mut inbox).await else {
            panic!("renderer command should win against a pending import");
        };
        assert!(matches!(command.payload, RendererCommand::CreateScene));

        scheduler.shutdown();
        assert!(scheduler.active.is_none());
        assert!(scheduler.pending.is_empty());
    }

    #[tokio::test]
    async fn queued_imports_complete_in_fifo_order_with_one_active_task() {
        let (_endpoint, mut inbox) = renderer_transport(4);
        let mut scheduler = AssetImportScheduler::default();
        let request_ids = [1, 2, 3].map(|raw| RequestId::from_raw(raw).unwrap());

        for request_id in request_ids {
            scheduler.enqueue(
                Some(request_id),
                std::env::temp_dir().join(format!(
                    "rebuild1-scheduler-missing-{}-{}.r1scene",
                    std::process::id(),
                    request_id.raw()
                )),
            );
        }

        scheduler.start_next_if_idle();
        assert!(scheduler.active.is_some());
        assert_eq!(scheduler.pending.len(), 2);

        for expected in request_ids {
            let RendererLoopEvent::AssetImported(completion) =
                scheduler.next_event(&mut inbox).await
            else {
                panic!("active import should complete before the idle command channel");
            };
            assert_eq!(completion.request_id, Some(expected));
            assert!(completion.result.is_err());
            assert!(scheduler.active.is_none());
        }

        assert!(scheduler.pending.is_empty());
        scheduler.shutdown();
    }
}
