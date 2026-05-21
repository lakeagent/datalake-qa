"""
Evaluation Framework for Data Lake Benchmark

This package provides:
- LLM Factory: Unified interface for OpenAI, Anthropic, and Google models
- Agent Tools: Functions for accessing the S3 data lake
- Agent Runner: Orchestration for running agents on tasks
- Metrics: Evaluation metrics for comparing answers

Usage:
    from evaluation import AgentRunner, LLMFactory

    # Run single question
    runner = AgentRunner(model="gpt-5.2")
    result = runner.run("What is the capital of France?")

    # Run batch evaluation
    from evaluation import BatchRunner
    batch = BatchRunner(model="gpt-5.2")
    results = batch.run_from_files(["tasks/task_1.json", "tasks/task_2.json"])
"""

from .llm.llm_factory import (
    LLMFactory,
    BaseLLM,
    OpenAILLM,
    BedrockLLM,
    DeepSeekLLM,
    LLMResponse,
    ToolCall,
    Provider,
    get_llm,
    complete,
)

from .agent_runner import (
    AgentRunner,
    BatchRunner,
    AgentResult,
    AgentConfig,
)

from .tools.tools_schema import (
    AGENT_TOOLS,
    get_tools_for_provider,
)

from .metrics import (
    compute_exact_match,
    compute_f1_score,
    compute_semantic_similarity,
    normalize_text,
)

__all__ = [
    # LLM Factory
    "LLMFactory",
    "BaseLLM",
    "OpenAILLM",
    "BedrockLLM",
    "LLMResponse",
    "ToolCall",
    "Provider",
    "get_llm",
    "complete",
    # Agent Runner
    "AgentRunner",
    "BatchRunner",
    "AgentResult",
    "AgentConfig",
    # Tools
    "AGENT_TOOLS",
    "get_tools_for_provider",
    # Metrics
    "compute_exact_match",
    "compute_f1_score",
    "compute_semantic_similarity",
    "normalize_text",
]
