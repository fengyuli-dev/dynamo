// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end check that the OTLP audit sink actually exports a record over the
//! wire under the *production* env config (DYN_AUDIT_SINKS=otel +
//! DYN_AUDIT_FORCE_LOGGING=true + OTEL_EXPORTER_OTLP_LOGS_ENDPOINT). Stands up a
//! mock OTLP/HTTP collector on localhost, runs the audit pipeline
//! (bus -> worker -> OtelSink -> opentelemetry-otlp batch exporter), forces a
//! flush on shutdown, and asserts the captured POST body carries the request id
//! and the serialized request/response payload (the `payload` log attribute).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::audit::{bus, config, handle, sink};
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};
use temp_env::async_with_vars;
use tokio_util::sync::CancellationToken;

const SENTINEL: &str = "OTEL-SENTINEL-deadbeef-9f1c";
const REQUEST_ID: &str = "rid-OTEL-SENTINEL-deadbeef-9f1c";

fn test_request() -> NvCreateChatCompletionRequest {
    let json = serde_json::json!({
        "model": "deepseek-ai/deepseek-v4-pro",
        "messages": [{"role": "user", "content": format!("hello {SENTINEL}")}],
        "store": true,
        "stream": false
    });
    serde_json::from_value(json).expect("request")
}

fn test_response() -> NvCreateChatCompletionResponse {
    let json = serde_json::json!({
        "id": "chatcmpl-otel",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "deepseek-ai/deepseek-v4-pro",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi back"},
            "finish_reason": "stop"
        }]
    });
    serde_json::from_value(json).expect("response")
}

/// Minimal one-shot OTLP/HTTP collector. Accepts a single connection, reads the
/// full HTTP request (honoring Content-Length), replies 200 with an empty
/// protobuf body (a valid ExportLogsServiceResponse), and ships the raw request
/// bytes back over a channel.
fn spawn_mock_collector() -> (u16, mpsc::Receiver<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            stream
                .set_read_timeout(Some(Duration::from_secs(8)))
                .ok();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let mut content_len: Option<usize> = None;
            let mut header_end: Option<usize> = None;
            loop {
                match stream.read(&mut tmp) {
                    Ok(0) => break,
                    Ok(n) => {
                        buf.extend_from_slice(&tmp[..n]);
                        if header_end.is_none()
                            && let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n")
                        {
                            header_end = Some(pos + 4);
                            let head = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                            for line in head.lines() {
                                if let Some(v) = line.strip_prefix("content-length:") {
                                    content_len = v.trim().parse::<usize>().ok();
                                }
                            }
                        }
                        if let (Some(he), Some(cl)) = (header_end, content_len)
                            && buf.len() >= he + cl
                        {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/x-protobuf\r\nContent-Length: 0\r\n\r\n",
            );
            let _ = stream.flush();
            let _ = tx.send(buf);
        }
    });
    (port, rx)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn otel_sink_exports_payload_over_the_wire() {
    let (port, rx) = spawn_mock_collector();
    let endpoint = format!("http://127.0.0.1:{port}/v1/logs");

    async_with_vars(
        [
            ("DYN_AUDIT_SINKS", Some("otel")),
            ("DYN_AUDIT_FORCE_LOGGING", Some("true")),
            ("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", Some(endpoint.as_str())),
            ("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", Some("http/protobuf")),
        ],
        async {
            // Sanity: the prod-style env makes audit + the otel sink active.
            assert!(config::policy().enabled, "audit should be enabled");
            assert!(
                config::otel_sink_capture_enabled(),
                "otel sink should be selected"
            );

            let shutdown = CancellationToken::new();
            bus::init(config::policy().capacity);
            sink::spawn_workers_from_env(shutdown.clone())
                .await
                .expect("spawn otel worker");
            tokio::time::sleep(Duration::from_millis(150)).await;

            // Emit one combined record exactly as the preprocessor does.
            let mut h = handle::create_handle(&test_request(), REQUEST_ID, None)
                .expect("force_logging should yield a handle even though this is a fresh request");
            h.set_request(Arc::new(test_request()));
            h.set_response(Arc::new(test_response()));
            h.emit();

            // Let the worker consume + enqueue, then force the batch flush.
            tokio::time::sleep(Duration::from_millis(400)).await;
            shutdown.cancel();

            // Wait (off the async runtime) for the collector to capture the POST.
            let body = tokio::task::spawn_blocking(move || {
                rx.recv_timeout(Duration::from_secs(12))
            })
            .await
            .expect("join blocking")
            .expect("OTLP collector received an export POST");

            let body_str = String::from_utf8_lossy(&body);
            assert!(
                body_str.contains("/v1/logs"),
                "POST should target the logs endpoint; got:\n{}",
                &body_str[..body_str.len().min(400)]
            );
            // The request id rides as the `rid` log attribute; the request +
            // response payload rides as the `payload` attribute (serialized
            // AuditRecord JSON). Both are UTF-8 strings inside the OTLP protobuf.
            assert!(
                body_str.contains(REQUEST_ID),
                "exported record should carry the request id"
            );
            assert!(
                body_str.contains(SENTINEL),
                "exported payload should carry the request content (payload attribute)"
            );
            assert!(
                body_str.contains("hi back"),
                "exported payload should carry the response content"
            );
        },
    )
    .await;
}
