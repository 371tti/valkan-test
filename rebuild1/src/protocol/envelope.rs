use super::{FrameId, ProtocolVersion, RequestId};

#[derive(Clone, Debug)]
pub struct MessageEnvelope<T> {
    pub protocol_version: ProtocolVersion,
    pub request_id: Option<RequestId>,
    pub frame_id: Option<FrameId>,
    pub payload: T,
}

impl<T> MessageEnvelope<T> {
    /// Wraps a payload with the current protocol version.
    pub fn new(payload: T) -> Self {
        Self {
            protocol_version: ProtocolVersion::current(),
            request_id: None,
            frame_id: None,
            payload,
        }
    }

    /// Attaches a request id used to match a later response event.
    pub fn with_request_id(mut self, request_id: RequestId) -> Self {
        self.request_id = Some(request_id);
        self
    }

    /// Attaches the frame id associated with this message.
    pub fn with_frame_id(mut self, frame_id: FrameId) -> Self {
        self.frame_id = Some(frame_id);
        self
    }
}
