// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Orchestrator that delegates narrow sub-tasks to a worker model through a
//! synthetic tool call, transparent to the client.
//!
//! The orchestrator answers every client-visible turn. It is offered one
//! additional tool (`delegate_task` by default). Calling it *alone* hands the
//! task to the worker model; the worker's answer is fed back to the
//! orchestrator as a tool result so it can review it and continue -- perhaps
//! delegating again, calling one of its own (client) tools, or answering
//! directly. None of this round trip is visible to the client: from its point
//! of view the orchestrator either answered, or made one of its usual tool
//! calls, just as it would with `passthrough`.
//!
//! Delegation is capped at `max_delegations` orchestrator round trips per
//! client-visible turn, so an orchestrator that never stops delegating fails
//! the request instead of looping forever.
//!
//! Every orchestrator call is buffered so its tool calls can be inspected
//! before the client sees anything, mirroring [`super::advisor_gate`]. A turn
//! that mixes `delegate_task` with any other tool call is rejected: the other
//! call belongs to the client's own harness, and this algorithm cannot serve
//! both a local delegation and a client round trip from the same turn.

use std::sync::Arc;

use futures::future::try_join_all;
use serde_json::{Value, json};
use switchyard_protocol::{
    AggLlmResponse, Category, ContentBlock, InstructionBlock, LlmRequest, LlmResponse, Message,
    ModelId, Request, Response, Role, ToolCall, ToolChoice, ToolDefinition, ToolResult,
};

use super::util::prompts::drop_exact_replay;
use crate::core::algorithm::{Algorithm, Driver};
use crate::{LibsyError, Result, RoutingOutcome};

/// Default name of the synthetic tool exposed to the orchestrator.
pub const DEFAULT_TOOL_NAME: &str = "delegate_task";

/// Default description shown to the orchestrator for the delegation tool.
pub const DEFAULT_TOOL_DESCRIPTION: &str = "Delegate a small, narrowly-scoped, mechanical sub-task \
to a fast worker model: repository searches, information extraction, test or log analysis, \
boilerplate generation, or a simple, well-specified code change. Do not delegate architecture \
decisions, ambiguous requirements, security-sensitive judgment calls, or anything needing broad \
reasoning. Supply all the context the worker needs -- it cannot ask follow-up questions -- and \
review its result before relying on it.";

/// Orchestrator round trips allowed per client-visible turn, bounding a
/// delegation loop that never converges.
const DEFAULT_MAX_DELEGATIONS: u32 = 6;

/// Configuration for [`Delegate`].
#[derive(Clone, Debug)]
pub struct DelegateConfig {
    /// Name of the synthetic tool exposed to the orchestrator.
    pub tool_name: String,
    /// Description of the synthetic tool exposed to the orchestrator.
    pub tool_description: String,
    /// System instruction prepended to a delegated sub-task's own request.
    pub worker_system_prompt: Option<String>,
    /// Orchestrator round trips allowed per client-visible turn.
    pub max_delegations: u32,
}

impl Default for DelegateConfig {
    fn default() -> Self {
        Self {
            tool_name: DEFAULT_TOOL_NAME.to_string(),
            tool_description: DEFAULT_TOOL_DESCRIPTION.to_string(),
            worker_system_prompt: None,
            max_delegations: DEFAULT_MAX_DELEGATIONS,
        }
    }
}

/// Routes every turn to the orchestrator, resolving its calls to the
/// delegation tool against the worker before the client ever sees them.
pub struct Delegate {
    config: DelegateConfig,
}

impl Delegate {
    /// Creates a delegation router.
    ///
    /// Returns an error when `tool_name` or `tool_description` is blank, or
    /// `max_delegations` is zero.
    pub fn new(config: DelegateConfig) -> Result<Self> {
        if config.tool_name.trim().is_empty() {
            return Err(algorithm_error("tool_name must not be empty"));
        }
        if config.tool_description.trim().is_empty() {
            return Err(algorithm_error("tool_description must not be empty"));
        }
        if config.max_delegations == 0 {
            return Err(algorithm_error("max_delegations must be at least 1"));
        }
        Ok(Self { config })
    }

    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.config.tool_name.clone(),
            description: Some(self.config.tool_description.clone()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The sub-task for the worker to perform, stated precisely."
                    },
                    "context": {
                        "type": "string",
                        "description": "All context the worker needs; it cannot ask follow-up questions."
                    },
                    "expected_output": {
                        "type": "string",
                        "description": "The shape of the answer you need back, e.g. \"JSON array\" or \"unified diff\"."
                    }
                },
                "required": ["task", "context"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    fn reject_reserved_tool_collision(&self, request: &Request) -> Result<()> {
        if request
            .llm_request
            .tools
            .iter()
            .any(|tool| tool.name == self.config.tool_name)
        {
            return Err(algorithm_error(format!(
                "request already defines reserved tool {:?}",
                self.config.tool_name
            )));
        }
        Ok(())
    }

    /// Tool calls in this turn's output, partitioned into delegate calls and
    /// every other (client-owned) tool call.
    fn partition_tool_calls<'a>(&self, agg: &'a AggLlmResponse) -> (Vec<&'a ToolCall>, usize) {
        let mut delegate_calls = Vec::new();
        let mut other = 0usize;
        for block in agg.outputs.iter().flat_map(|output| &output.content) {
            if let ContentBlock::ToolCall(tool_call) = block {
                if tool_call.name == self.config.tool_name {
                    delegate_calls.push(tool_call);
                } else {
                    other += 1;
                }
            }
        }
        (delegate_calls, other)
    }

    fn worker_request(&self, base: &Request, tool_call: &ToolCall) -> Result<Request> {
        let task = string_field(&tool_call.arguments, "task").ok_or_else(|| {
            algorithm_error(format!(
                "{} call {} requires a non-empty string \"task\"",
                self.config.tool_name, tool_call.id
            ))
        })?;
        let context = string_field(&tool_call.arguments, "context").unwrap_or_default();
        let expected_output = string_field(&tool_call.arguments, "expected_output");

        let mut prompt = task;
        if !context.is_empty() {
            prompt.push_str("\n\nContext:\n");
            prompt.push_str(&context);
        }
        if let Some(expected_output) = expected_output {
            prompt.push_str("\n\nRespond with: ");
            prompt.push_str(&expected_output);
        }

        let instructions = match &self.config.worker_system_prompt {
            Some(system_prompt) => vec![InstructionBlock {
                role: Role::System,
                content: vec![ContentBlock::Text {
                    text: system_prompt.clone(),
                }],
            }],
            None => Vec::new(),
        };

        Ok(Request {
            llm_request: LlmRequest {
                instructions,
                messages: vec![Message::text(Role::User, prompt)],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: base.metadata.clone(),
        })
    }
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn worker_text(agg: &AggLlmResponse) -> String {
    agg.outputs
        .iter()
        .flat_map(|output| &output.content)
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn algorithm_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: message.into(),
    }
}

/// Buffers a call's response, returning its served model and aggregate.
async fn buffer(
    served_fallback: &ModelId,
    response: Response,
) -> Result<(ModelId, AggLlmResponse)> {
    let served = response
        .served_model()
        .cloned()
        .unwrap_or_else(|| served_fallback.clone());
    let Response {
        llm_response,
        metadata: _,
        upstream_headers: _,
    } = response;
    let agg = llm_response
        .into_agg()
        .await
        .map_err(|error| LibsyError::client_call(served.clone(), error))?;
    Ok((served, agg))
}

#[async_trait::async_trait]
impl Algorithm for Delegate {
    fn name(&self) -> &str {
        "delegate"
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        self.reject_reserved_tool_collision(&request)?;

        let orchestrator_models = driver.models_for(&Category::Capable).to_vec();
        let orchestrator = orchestrator_models
            .first()
            .ok_or_else(|| algorithm_error("no models available for category Capable"))?
            .clone();
        let worker_models = driver.models_for(&Category::Efficient).to_vec();
        if worker_models.is_empty() {
            return Err(algorithm_error(
                "no models available for category Efficient",
            ));
        }

        let stream_requested = request.llm_request.stream;
        request.llm_request.tools.push(self.tool_definition());
        request
            .llm_request
            .tool_choice
            .get_or_insert(ToolChoice::Auto);
        // Every orchestrator call in this loop is buffered so its tool calls can be
        // inspected before anything reaches the client.
        request.llm_request.stream = false;
        drop_exact_replay(&mut request);

        for round in 0..self.config.max_delegations {
            let response = driver
                .call_model(request.clone(), orchestrator_models.clone())
                .await?;
            let (served, agg) = buffer(&orchestrator, response).await?;

            let (delegate_calls, other_tool_calls) = self.partition_tool_calls(&agg);
            if delegate_calls.is_empty() {
                driver.set_evidence(json!({"source": "delegate", "rounds": round}));
                return Ok(RoutingOutcome::answered(
                    served,
                    request,
                    finish(agg, stream_requested),
                ));
            }
            if other_tool_calls > 0 {
                return Err(algorithm_error(format!(
                    "{} must be called alone; the orchestrator mixed it with {other_tool_calls} \
                     other tool call(s)",
                    self.config.tool_name
                )));
            }

            tracing::info!(
                tool = %self.config.tool_name,
                round,
                delegations = delegate_calls.len(),
                "delegating sub-task(s) to worker"
            );

            let assistant_content = agg
                .first_output()
                .map(|output| output.content.clone())
                .unwrap_or_default();
            request.llm_request.messages.push(Message {
                role: Role::Assistant,
                content: assistant_content,
            });

            let worker_calls = delegate_calls.iter().map(|tool_call| {
                self.dispatch_to_worker(&driver, &request, &worker_models, tool_call)
            });
            let results = try_join_all(worker_calls).await?;
            request.llm_request.messages.extend(results);
            drop_exact_replay(&mut request);
        }

        Err(algorithm_error(format!(
            "orchestrator delegated more than max_delegations ({}) times in one turn",
            self.config.max_delegations
        )))
    }
}

impl Delegate {
    /// Runs one delegated sub-task and returns its result as a tool-result message.
    async fn dispatch_to_worker(
        &self,
        driver: &Driver,
        base: &Request,
        worker_models: &[ModelId],
        tool_call: &ToolCall,
    ) -> Result<Message> {
        let worker_request = self.worker_request(base, tool_call)?;
        let response = driver
            .call_model(worker_request, worker_models.to_vec())
            .await?;
        let (served, agg) = buffer(&worker_models[0], response).await?;
        tracing::debug!(worker = %served, tool_call_id = %tool_call.id, "worker delegation complete");
        Ok(Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: vec![ContentBlock::Text {
                    text: worker_text(&agg),
                }],
                is_error: None,
            })],
        })
    }
}

fn finish(agg: AggLlmResponse, stream: bool) -> Response {
    Response {
        llm_response: if stream {
            LlmResponse::Stream(agg.into_stream())
        } else {
            LlmResponse::Agg(agg)
        },
        metadata: None,
        upstream_headers: http::HeaderMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use serde_json::json;
    use switchyard_protocol::{
        LlmRequest, Message, Metadata, Request, ResponseOutput, Role, StopReason, ToolCall,
        completion_text, text_response,
    };

    use super::*;
    use crate::RuntimeModels;
    use crate::core::testing::test_drive_with_models;

    const ORCHESTRATOR: &str = "model/orchestrator";
    const WORKER: &str = "model/worker";

    fn algorithm(config: DelegateConfig) -> Arc<dyn Algorithm> {
        Arc::new(Delegate::new(config).expect("config should be valid"))
    }

    fn models() -> RuntimeModels {
        RuntimeModels::new(HashMap::from([
            (Category::Capable, vec![ModelId::from(ORCHESTRATOR)]),
            (Category::Efficient, vec![ModelId::from(WORKER)]),
        ]))
    }

    fn request(stream: bool) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("switchyard/delegate".to_string()),
                messages: vec![Message::text(Role::User, "Summarize the failing tests")],
                stream,
                ..LlmRequest::default()
            },
            metadata: Some(Metadata::default()),
            ..Request::default()
        }
    }

    fn tool_call_response(tool_name: &str, arguments: serde_json::Value) -> Response {
        Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                outputs: vec![ResponseOutput {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall(ToolCall {
                        id: "call-1".to_string(),
                        name: tool_name.to_string(),
                        arguments,
                    })],
                    url_citations: Vec::new(),
                    stop_reason: Some(StopReason::ToolUse),
                }],
                ..AggLlmResponse::default()
            }),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }

    fn text_reply(text: &str) -> Response {
        Response {
            llm_response: LlmResponse::Agg(text_response(None, text)),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }

    #[tokio::test]
    async fn plain_answer_passes_through_after_tool_injection() {
        let captured = Arc::new(Mutex::new(None));
        let capture = Arc::clone(&captured);
        let (selected, response) = test_drive_with_models(
            algorithm(DelegateConfig::default()),
            request(false),
            models(),
            move |target, request| {
                let capture = Arc::clone(&capture);
                async move {
                    *capture.lock().expect("lock") = Some(request);
                    Ok(text_reply_from(target, "plain answer"))
                }
            },
        )
        .await
        .expect("routing should succeed");

        assert_eq!(selected, ORCHESTRATOR);
        let sent = captured.lock().expect("lock").take().expect("request");
        assert_eq!(sent.llm_request.tools.len(), 1);
        assert_eq!(sent.llm_request.tools[0].name, DEFAULT_TOOL_NAME);
        assert_eq!(sent.llm_request.tool_choice, Some(ToolChoice::Auto));
        assert!(!sent.llm_request.stream);

        let agg = response
            .llm_response
            .into_agg()
            .await
            .expect("aggregating test response");
        assert_eq!(completion_text(&agg), "plain answer");
    }

    #[tokio::test]
    async fn delegated_task_result_is_fed_back_and_reviewed() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&calls);
        let (selected, response) = test_drive_with_models(
            algorithm(DelegateConfig::default()),
            request(false),
            models(),
            move |target: ModelId, request| {
                let captured = Arc::clone(&captured);
                async move {
                    captured
                        .lock()
                        .expect("lock")
                        .push((target.to_string(), request));
                    match target.as_str() {
                        ORCHESTRATOR if captured.lock().expect("lock").len() == 1 => {
                            Ok(tool_call_response(
                                DEFAULT_TOOL_NAME,
                                json!({"task": "list failing tests", "context": "pytest output: ..."}),
                            ))
                        }
                        WORKER => Ok(text_reply("3 tests failed: test_a, test_b, test_c")),
                        ORCHESTRATOR => Ok(text_reply("Three tests are failing: test_a/b/c.")),
                        other => panic!("unexpected target {other}"),
                    }
                }
            },
        )
        .await
        .expect("routing should succeed");

        assert_eq!(selected, ORCHESTRATOR);
        {
            let calls = calls.lock().expect("lock");
            assert_eq!(
                calls
                    .iter()
                    .map(|(target, _)| target.as_str())
                    .collect::<Vec<_>>(),
                [ORCHESTRATOR, WORKER, ORCHESTRATOR]
            );

            // The worker never saw the orchestrator's tool-call plumbing, just the task.
            let worker_request = &calls[1].1;
            assert_eq!(worker_request.llm_request.tools.len(), 0);
            assert_eq!(
                worker_request.llm_request.messages[0].text_content(""),
                Some("list failing tests\n\nContext:\npytest output: ...".to_string())
            );

            // The orchestrator's second call saw the delegated turn and the worker's result.
            let followup_request = &calls[2].1;
            let messages = &followup_request.llm_request.messages;
            assert!(matches!(
                messages[messages.len() - 2].content[0],
                ContentBlock::ToolCall(_)
            ));
            assert!(matches!(
                messages[messages.len() - 1].content[0],
                ContentBlock::ToolResult(_)
            ));
        }

        let agg = response
            .llm_response
            .into_agg()
            .await
            .expect("aggregating test response");
        assert_eq!(
            completion_text(&agg),
            "Three tests are failing: test_a/b/c."
        );
    }

    #[tokio::test]
    async fn other_tool_calls_pass_through_untouched() {
        let (selected, response) = test_drive_with_models(
            algorithm(DelegateConfig::default()),
            request(false),
            models(),
            move |target: ModelId, _request| async move {
                match target.as_str() {
                    ORCHESTRATOR => Ok(tool_call_response("read_file", json!({"path": "a.rs"}))),
                    other => panic!("unexpected target {other}"),
                }
            },
        )
        .await
        .expect("routing should succeed");

        assert_eq!(selected, ORCHESTRATOR);
        let agg = response
            .llm_response
            .into_agg()
            .await
            .expect("aggregating test response");
        assert!(matches!(
            agg.outputs[0].content[0],
            ContentBlock::ToolCall(ref call) if call.name == "read_file"
        ));
    }

    #[tokio::test]
    async fn mixing_delegate_task_with_another_tool_call_is_rejected() {
        let algorithm = algorithm(DelegateConfig::default());
        let mut response =
            tool_call_response(DEFAULT_TOOL_NAME, json!({"task": "x", "context": "y"}));
        let LlmResponse::Agg(agg) = &mut response.llm_response else {
            unreachable!()
        };
        agg.outputs[0]
            .content
            .push(ContentBlock::ToolCall(ToolCall {
                id: "call-2".to_string(),
                name: "shell".to_string(),
                arguments: json!({"command": "pwd"}),
            }));
        let response = Mutex::new(Some(response));

        let result = test_drive_with_models(algorithm, request(false), models(), move |_, _| {
            let response = response.lock().expect("lock").take();
            async move { Ok(response.expect("one call")) }
        })
        .await;

        assert!(matches!(result, Err(LibsyError::AlgorithmError { .. })));
    }

    #[tokio::test]
    async fn rejects_reserved_tool_collision() {
        let mut req = request(false);
        req.llm_request.tools.push(ToolDefinition {
            name: DEFAULT_TOOL_NAME.to_string(),
            description: None,
            parameters: json!({}),
            strict: None,
        });
        let algorithm = algorithm(DelegateConfig::default());
        let result = test_drive_with_models(algorithm, req, models(), |_, _| async move {
            unreachable!("collision must fail before a model call")
        })
        .await;

        assert!(matches!(
            result,
            Err(LibsyError::AlgorithmError { message }) if message.contains("reserved tool")
        ));
    }

    #[tokio::test]
    async fn rejects_a_delegation_without_a_task() {
        let algorithm = algorithm(DelegateConfig::default());
        let response = Mutex::new(Some(tool_call_response(
            DEFAULT_TOOL_NAME,
            json!({"context": "no task field"}),
        )));
        let result = test_drive_with_models(algorithm, request(false), models(), move |_, _| {
            let response = response.lock().expect("lock").take();
            async move { Ok(response.expect("one call")) }
        })
        .await;

        assert!(matches!(result, Err(LibsyError::AlgorithmError { .. })));
    }

    #[tokio::test]
    async fn a_loop_that_never_converges_fails_at_the_cap() {
        let algorithm = algorithm(DelegateConfig {
            max_delegations: 2,
            ..DelegateConfig::default()
        });
        let result = test_drive_with_models(
            algorithm,
            request(false),
            models(),
            move |target: ModelId, _| async move {
                match target.as_str() {
                    ORCHESTRATOR => Ok(tool_call_response(
                        DEFAULT_TOOL_NAME,
                        json!({"task": "keep going", "context": "..."}),
                    )),
                    WORKER => Ok(text_reply("ok")),
                    other => panic!("unexpected target {other}"),
                }
            },
        )
        .await;

        assert!(matches!(result, Err(LibsyError::AlgorithmError { .. })));
    }

    fn text_reply_from(target: ModelId, text: &str) -> Response {
        Response {
            llm_response: LlmResponse::Agg(text_response(Some(target.to_string()), text)),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }
}
