// VeloxLLM HTTP Server
// OpenAI-compatible API server with streaming support.

use anyhow::Result;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{info, error};

use crate::inference::VeloxEngine;
use crate::sampler::SamplerConfig;

/// Start the HTTP server
pub async fn start_server(
    engine: VeloxEngine,
    host: &str,
    port: u16,
    max_concurrent: usize,
) -> Result<()> {
    let engine = Arc::new(engine);
    let addr = format!("{}:{}", host, port);

    let listener = TcpListener::bind(&addr).await?;
    info!("Server listening on {}", addr);

    loop {
        let (mut socket, addr) = listener.accept().await?;
        let engine = engine.clone();

        tokio::spawn(async move {
            let mut buf = vec![0u8; 65536];

            loop {
                let n = match socket.read(&mut buf).await {
                    Ok(0) => return,
                    Ok(n) => n,
                    Err(e) => {
                        error!("Read error from {}: {}", addr, e);
                        return;
                    }
                };

                let request = String::from_utf8_lossy(&buf[..n]);

                // Parse HTTP request (simplified)
                let response = handle_request(&engine, &request).await;

                if let Err(e) = socket.write_all(response.as_bytes()).await {
                    error!("Write error to {}: {}", addr, e);
                    return;
                }
            }
        });
    }
}

/// Handle an HTTP request
async fn handle_request(engine: &VeloxEngine, request: &str) -> String {
    // Parse request line
    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();

    if parts.len() < 2 {
        return http_response(400, "Bad Request");
    }

    let method = parts[0];
    let path = parts[1];

    match (method, path) {
        ("GET", "/") => {
            http_json_response(200, r#"{"status":"ok","engine":"veloxllm","version":"0.9.0"}"#)
        }
        ("GET", "/v1/models") => {
            http_json_response(200, r#"{
                "object": "list",
                "data": [{
                    "id": "veloxllm",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "veloxllm"
                }]
            }"#)
        }
        ("POST", "/v1/chat/completions") => {
            // Extract body from request
            let body_start = request.find("\r\n\r\n").unwrap_or(0) + 4;
            let body = &request[body_start..];

            // Parse JSON body (simplified)
            let prompt = extract_prompt_from_body(body);

            match engine.generate(&prompt, 256, SamplerConfig {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
            }).await {
                Ok(tokens) => {
                    let text = engine.tokenizer.decode(&tokens).unwrap_or_default();
                    let response = format!(r#"{{
                        "id": "chatcmpl-veloxllm",
                        "object": "chat.completion",
                        "created": {},
                        "model": "veloxllm",
                        "choices": [{{
                            "index": 0,
                            "message": {{
                                "role": "assistant",
                                "content": {}
                            }},
                            "finish_reason": "stop"
                        }}],
                        "usage": {{
                            "prompt_tokens": {},
                            "completion_tokens": {},
                            "total_tokens": {}
                        }}
                    }}"#,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        serde_json::to_string(&text).unwrap_or_default(),
                        prompt.len(),
                        tokens.len(),
                        prompt.len() + tokens.len(),
                    );
                    http_json_response(200, &response)
                }
                Err(e) => {
                    http_json_response(500, &format!(r#"{{"error": "{}"}}"#, e))
                }
            }
        }
        ("POST", "/v1/completions") => {
            let body_start = request.find("\r\n\r\n").unwrap_or(0) + 4;
            let body = &request[body_start..];

            let prompt = extract_prompt_from_body(body);

            match engine.generate(&prompt, 256, SamplerConfig {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
            }).await {
                Ok(tokens) => {
                    let text = engine.tokenizer.decode(&tokens).unwrap_or_default();
                    let response = format!(r#"{{
                        "id": "cmpl-veloxllm",
                        "object": "text_completion",
                        "created": {},
                        "model": "veloxllm",
                        "choices": [{{
                            "text": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }}]
                    }}"#,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        serde_json::to_string(&text).unwrap_or_default(),
                    );
                    http_json_response(200, &response)
                }
                Err(e) => {
                    http_json_response(500, &format!(r#"{{"error": "{}"}}"#, e))
                }
            }
        }
        _ => http_response(404, "Not Found"),
    }
}

fn extract_prompt_from_body(body: &str) -> String {
    // Simplified JSON parsing
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
        // Try chat completions format
        if let Some(messages) = json.get("messages").and_then(|m| m.as_array()) {
            let mut prompt = String::new();
            for msg in messages {
                if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                    prompt.push_str(content);
                    prompt.push('\n');
                }
            }
            return prompt;
        }
        // Try completions format
        if let Some(prompt) = json.get("prompt").and_then(|p| p.as_str()) {
            return prompt.to_string();
        }
    }
    "Hello".to_string()
}

fn http_response(status: u16, body: &str) -> String {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };

    format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n{}",
        status, status_text, body.len(), body
    )
}

fn http_json_response(status: u16, body: &str) -> String {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };

    format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
        status, status_text, body.len(), body
    )
}
