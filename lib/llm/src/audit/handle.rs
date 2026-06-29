// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use axum::http::HeaderMap;

use super::{bus, config};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

/// Context key under which the HTTP layer stashes the inbound request headers
/// for the OTLP audit sink to export. Only populated when the `otel` sink is
/// active (see `config::otel_sink_capture_enabled`).
pub const OTEL_HTTP_HEADERS_CONTEXT_KEY: &str = "audit.otel.http.request.headers";

/// Captured inbound HTTP request headers, forwarded from the HTTP entrypoint to
/// the audit record so the OTLP sink can attach (redacted) header attributes.
#[derive(Clone)]
pub struct AuditHttpRequestHeaders {
    headers: Arc<HeaderMap>,
}

impl AuditHttpRequestHeaders {
    pub fn new(headers: Arc<HeaderMap>) -> Self {
        Self { headers }
    }

    pub fn headers(&self) -> &HeaderMap {
        self.headers.as_ref()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AuditRecord {
    pub schema_version: u32,
    pub request_id: String,
    pub requested_streaming: bool,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Arc<NvCreateChatCompletionRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Arc<NvCreateChatCompletionResponse>>,
    /// Inbound HTTP headers captured for the OTLP sink. Never serialized into
    /// the JSONL/NATS/stderr payloads — the otel sink injects a redacted view
    /// into its own `payload.http.request` field.
    #[serde(skip)]
    pub otel_http_headers: Option<Arc<AuditHttpRequestHeaders>>,
}

pub struct AuditHandle {
    requested_streaming: bool,
    request_id: String,
    model: String,
    req_full: Option<Arc<NvCreateChatCompletionRequest>>,
    resp_full: Option<Arc<NvCreateChatCompletionResponse>>,
    otel_http_headers: Option<Arc<AuditHttpRequestHeaders>>,
}

impl AuditHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }

    pub fn set_request(&mut self, req: Arc<NvCreateChatCompletionRequest>) {
        self.req_full = Some(req);
    }
    pub fn set_response(&mut self, resp: Arc<NvCreateChatCompletionResponse>) {
        self.resp_full = Some(resp);
    }

    /// Emit exactly once (publishes to the bus; sinks do I/O).
    pub fn emit(self) {
        let rec = AuditRecord {
            schema_version: 1,
            request_id: self.request_id,
            requested_streaming: self.requested_streaming,
            model: self.model,
            request: self.req_full,
            response: self.resp_full,
            otel_http_headers: self.otel_http_headers,
        };
        bus::publish(rec);
    }
}

pub fn create_handle(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    otel_http_headers: Option<Arc<AuditHttpRequestHeaders>>,
) -> Option<AuditHandle> {
    let policy = config::policy();
    create_handle_with_config(
        req,
        request_id,
        policy.enabled,
        policy.force_logging,
        otel_http_headers,
    )
}

fn create_handle_with_config(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    enabled: bool,
    force_logging: bool,
    otel_http_headers: Option<Arc<AuditHttpRequestHeaders>>,
) -> Option<AuditHandle> {
    if !enabled {
        return None;
    }
    // If force_logging is enabled, ignore the store flag
    if !force_logging && !req.inner.store.unwrap_or(false) {
        return None;
    }
    let requested_streaming = req.inner.stream.unwrap_or(false);
    let model = req.inner.model.clone();

    Some(AuditHandle {
        requested_streaming,
        request_id: request_id.to_string(),
        model,
        req_full: None,
        resp_full: None,
        otel_http_headers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn create_test_request(model: &str, store: bool) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "store": store
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    fn create_test_request_with_nvext() -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "store": true,
            "nvext": {
                "agent_hints": {
                    "priority": 5
                }
            }
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    fn create_test_response(content: &str) -> NvCreateChatCompletionResponse {
        let json = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }]
        });
        serde_json::from_value(json).expect("Failed to create test response")
    }

    /// Test that DYN_AUDIT_FORCE_LOGGING=true bypasses store=false
    /// When force logging is enabled, audit handle should be created even when store=false
    #[test]
    fn test_force_logging_bypasses_store() {
        let request = create_test_request("test-model", false);
        let handle = create_handle_with_config(&request, "test-id", true, true, None);

        assert!(
            handle.is_some(),
            "force logging should create a handle even with store=false"
        );
    }

    #[test]
    fn audit_record_serializes_nvext_and_response_content() {
        let record = AuditRecord {
            schema_version: 1,
            request_id: "req-123".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            request: Some(Arc::new(create_test_request_with_nvext())),
            response: Some(Arc::new(create_test_response("final answer"))),
            otel_http_headers: None,
        };

        let value = serde_json::to_value(record).unwrap();

        assert_eq!(value["request"]["nvext"]["agent_hints"]["priority"], 5);
        assert_eq!(
            value["response"]["choices"][0]["message"]["content"],
            "final answer"
        );
    }
}
