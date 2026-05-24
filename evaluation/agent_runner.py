"""
Agent Runner - Executes LLM agents with tool calling for data lake tasks.

This module orchestrates:
1. LLM calls (via llm_factory)
2. Tool execution (via agent_tools)
3. Conversation management
4. Answer extraction

Usage:
    from agent_runner import AgentRunner

    runner = AgentRunner(model="gpt-5.2")
    result = runner.run("What is the population of California according to the 2020 census?")
"""

import json
import time
import logging
import os
import re
import tempfile
import shutil
import multiprocessing
import queue
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from .llm.llm_factory import LLMFactory, LLMResponse, ToolCall
from .tools import agent_tools


def _parse_tool_call_from_text(content: str) -> Optional[ToolCall]:
    """
    Fallback parser for models that output tool calls as plain text JSON.
    Handles formats like: {"type": "function", "name": "...", "parameters": {...}}
    or: {"name": "...", "parameters": {...}}
    """
    if not content:
        return None

    # Try to find JSON in the content
    text = content.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Try to parse as JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from text
        match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(data, dict):
        return None

    # Extract tool name and parameters
    name = data.get("name")
    if not name:
        return None

    # Parameters can be in "parameters" or "arguments"
    params = data.get("parameters", data.get("arguments", {}))
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {}

    return ToolCall(
        id=f"text_parsed_{int(time.time()*1000)}",
        name=name,
        arguments=params if isinstance(params, dict) else {}
    )

DEFAULT_SYSTEM_PROMPT = """You are a data analysis agent working with PUBLIC GOVERNMENT DATASETS (data.gov, census, etc.).

## HOW THIS WORKS - READ CAREFULLY
This is an INTERACTIVE system. You output ONE tool call, then STOP. The system executes your tool and returns the REAL result. Then you are called again to pick the next tool.

DO NOT:
- Output multiple tool calls
- Simulate or hallucinate results (e.g., {"result": ...} or {"error": ...})
- Continue the conversation yourself
- Make up data

ONLY output a single JSON object with your tool call. The system handles execution.

## DATA ACCESS (each step = one tool call)
- search(prefixes) or search_keyword(keywords) → returns dataset_ids (identifiers, not data)
  - Pass a list of strings: search(["climate", "weather"]) or search_keyword(["police", "crime"])
- list_files(dataset_ids) → returns file paths in datasets
  - Pass a list of strings: list_files(["dataset1", "dataset2"])
- download(files) → downloads files to sandbox (max 5 per call)
  - Pass a list of {dataset_id, file_path}: download([{"dataset_id": "...", "file_path": "..."}])
- execute_code(code) → runs Python on downloaded files (use print()!)
- submit_answer() → when ready to submit final answer

## CRITICAL: VERIFY DATA SOURCES
Dataset names can be misleading! Example: "traffic-incidents-2020" could be from Chicago, NYC, or any other city.
ALWAYS check metadata before using a dataset:
- Download and read metadata files (e.g., metadata.json, catalog.txt) to find the actual source
- Look for: publisher, source, city/state, geographic coverage, agency name
- Verify the data matches what the question asks for (correct city, agency, time period)
- Two datasets with similar names may cover completely different locations!

## TIPS
- use search_keyword for semantic matching, SINGLE word preferred
- Always print() in execute_code to see output
- Check actual column names and date formats in the data
- Use full dataset for final answer, not just samples
- Answer format: [value] only, no labels or units

## TURN AND TIME LIMITS
- You have LIMITED TURNS. The system will show you remaining turns.
- There is also a TIME LIMIT. Do not waste time on excessive exploration.
- After inspecting files, IMMEDIATELY run execute_code to analyze data.
- If running low on turns or time, prioritize submitting your best answer."""


# Tool schemas for native tool calling (works with both OpenAI and Claude/Bedrock)
TOOL_SCHEMAS = [
    {
        "name": "search",
        "description": "Find datasets by name prefixes. Returns dataset identifiers (not data). Search multiple prefixes at once.",
        "parameters": {
            "type": "object",
            "properties": {
                "prefixes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of prefixes to search for (e.g., ['austin-police', 'climate', 'weather'])"
                }
            },
            "required": ["prefixes"]
        }
    },
    {
        "name": "search_keyword",
        "description": "Semantic keyword search across datasets. Search multiple keywords at once. Returns ranked dataset identifiers.",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of keywords to search for (e.g., ['police', 'crime', 'traffic'])"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 20)",
                    "default": 20
                }
            },
            "required": ["keywords"]
        }
    },
    {
        "name": "list_files",
        "description": "List files within datasets. Use dataset_ids from search results. List multiple datasets at once.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of dataset identifiers from search results (e.g., ['Barack_Obama', 'climate-data'])"
                }
            },
            "required": ["dataset_ids"]
        }
    },
    {
        "name": "download",
        "description": "Download files from datasets to the local sandbox for analysis. Max 5 files per call.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dataset_id": {"type": "string", "description": "Dataset identifier"},
                            "file_path": {"type": "string", "description": "Path to file within the dataset"}
                        },
                        "required": ["dataset_id", "file_path"]
                    },
                    "description": "List of files to download (max 5). Each with dataset_id and file_path.",
                    "maxItems": 5
                }
            },
            "required": ["files"]
        }
    },
    {
        "name": "inspect_file",
        "description": "Inspect a file to see its structure, columns, and sample data without downloading.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "Dataset identifier"
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to file within the dataset"
                }
            },
            "required": ["dataset_id", "file_path"]
        }
    },
    {
        "name": "execute_code",
        "description": "Execute Python code to analyze downloaded files. Use print() to see output. pandas, json, pathlib are available. IMPORTANT: Only use files you have successfully downloaded - use the exact local_path returned by the download tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use the exact local_path from download results. Do NOT guess file paths."
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "submit_answer",
        "description": "Submit your final answer when you have computed the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer. Format: just the value, e.g., '12345' not '12345 incidents'"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of how you computed the answer"
                }
            },
            "required": ["answer"]
        }
    }
]


# =============================================================================
# Logging Setup
# =============================================================================

# Global logger state is per-process
_logger: Optional[logging.Logger] = None
_log_run_id: Optional[str] = None
_log_model: Optional[str] = None
_log_batch: Optional[str] = None


def _slugify(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())


def _build_log_file(log_dir: str, run_id: Optional[str], model: Optional[str], batch: Optional[str]) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_part = _slugify(model)
    batch_part = _slugify(batch)
    parts = ["agent"]
    if model_part:
        parts.append(model_part)
    if batch_part:
        parts.append(batch_part)
    parts.append(run_id or timestamp)
    parts.append(f"pid{os.getpid()}")
    filename = "_".join(parts) + ".log"
    return os.path.join(log_dir, filename)


def setup_logger(log_dir: str = "logs", run_id: Optional[str] = None, model: Optional[str] = None, batch: Optional[str] = None) -> logging.Logger:
    """Setup logger that writes to file with full details.

    - File name includes run_id and pid to avoid collisions across processes.
    - Propagation is disabled to prevent duplicate writes via root handlers.
    """
    log_file = _build_log_file(log_dir, run_id, model, batch)

    logger = logging.getLogger(f"agent_runner_pid{os.getpid()}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")
    return logger


def get_logger(run_id: Optional[str] = None, model: Optional[str] = None, batch: Optional[str] = None) -> logging.Logger:
    """Get or create the per-process logger."""
    global _logger, _log_run_id, _log_model, _log_batch
    if run_id:
        _log_run_id = run_id
    if model:
        _log_model = model
    if batch:
        _log_batch = batch
    if _logger is None:
        _logger = setup_logger(run_id=_log_run_id, model=_log_model, batch=_log_batch)
    return _logger


def _clean_answer(answer) -> str:
    """Extract a concise answer from bracketed or verbose responses."""
    if not answer:
        return ""

    # Handle dict/list inputs (LLM sometimes outputs nested structures)
    if isinstance(answer, dict):
        # Try to extract answer from common keys
        answer = answer.get("answer") or answer.get("value") or answer.get("result") or str(answer)
    if not isinstance(answer, str):
        answer = str(answer)

    text = answer.strip()

    # Prefer bracketed content if present: [Answer]
    bracket_match = re.search(r"\[([^\[\]]+)\]", text)
    if bracket_match:
        text = bracket_match.group(1).strip()

    # Remove common answer prefixes
    text = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)

    # Strip surrounding quotes if present
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # If exactly one number appears, return just that number
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    numbers = [n.replace(",", "").rstrip(".") for n in numbers if re.search(r"\d", n)]
    if len(numbers) == 1:
        return numbers[0]

    return text


def _normalize_dataset_path(dataset_path: str) -> str:
    if not dataset_path:
        return ""
    path = dataset_path.rstrip("/")
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ""
    if parts[0] in ("datagov", "wikipedia"):
        parts = parts[1:]
    return parts[0] if parts else ""


def _create_isolated_sandbox(task_id: str) -> Path:
    """Create an isolated sandbox directory for a task."""
    sandbox_base = Path(__file__).resolve().parent.parent / ".sandbox_isolated"
    sandbox_base.mkdir(parents=True, exist_ok=True)
    
    task_sandbox = sandbox_base / f"task_{task_id}_{int(time.time() * 1000)}"
    task_sandbox.mkdir(parents=True, exist_ok=True)
    
    return task_sandbox


def _cleanup_isolated_sandbox(sandbox_path: Path) -> None:
    """Clean up an isolated sandbox directory."""
    if sandbox_path and sandbox_path.exists():
        try:
            shutil.rmtree(sandbox_path)
        except Exception as e:
            # Silently fail on cleanup - don't block the process
            pass


def _run_task_worker(
    task: Dict[str, Any],
    task_index: int,
    model: str,
    config: "AgentConfig",
    log_dir: str,
    run_id: str,
    batch_name: Optional[str],
    sandbox_path_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Worker function to run a single task in isolation.
    
    This function is designed to run in a separate process with its own:
    - Sandbox directory
    - Logger instance
    - LLM connection
    
    Args:
        task: Task dict with 'question' and optionally 'answer'
        task_index: Index of task in batch
        model: Model name
        config: Agent configuration
        log_dir: Directory for logs
    
    Returns:
        Result dict
    """
    from .metrics import compute_exact_match, compute_f1_score, normalize_text
    
    # Create isolated sandbox for this task and pin the sandbox used by agent_tools.
    # BatchRunner may pass a parent-created sandbox so it can clean up after a
    # hard process kill on timeout.
    sandbox_path = Path(sandbox_path_str) if sandbox_path_str else _create_isolated_sandbox(str(task_index))
    sandbox_path.mkdir(parents=True, exist_ok=True)
    agent_tools.set_sandbox_dir(sandbox_path)

    # Set up per-process logger (shared across tasks in this process)
    global _logger, _log_run_id, _log_model, _log_batch
    _logger = None  # reset in this process
    _log_run_id = run_id
    _log_model = model
    _log_batch = batch_name
    logger = get_logger(run_id=run_id, model=model, batch=batch_name)
    
    try:
        logger.info(f"Starting task {task_index + 1}: {task.get('question', '')[:80]}...")
        
        # Create and run agent
        runner = AgentRunner(model=model, config=config)
        result = runner.run(task["question"])
        
        result_dict = {
            "task_id": task.get("id", task_index),
            "model": model,
            "question": task.get("question", ""),
            "ground_truth": task.get("answer", ""),
            "predicted_answer": result.answer,
            "reasoning": result.reasoning,
            "sources_used": result.sources,
            "datasets_used": result.datasets_used,
            "datasets_executed": result.datasets_executed,
            "datasets_discovered": result.datasets_discovered,
            "tool_calls": result.tool_calls_made,
            "tokens": result.total_tokens,
            "input_tokens": result.input_tokens,
            "cached_input_tokens": result.cached_input_tokens,
            "cache_write_input_tokens": result.cache_write_input_tokens,
            "output_tokens": result.output_tokens,
            "agent_cost_usd": result.total_cost,
            "cost": result.total_cost,
            "cost_source": result.cost_source,
            "cost_note": result.cost_note,
            "cost_breakdown": result.cost_breakdown,
            "time": result.elapsed_time,
            "success": result.success,
            "error": result.error,
        }
        
        # Compute metrics if ground truth available
        if task.get("answer"):
            gt = str(task["answer"])
            pred = result.answer
            
            result_dict["exact_match"] = compute_exact_match(pred, gt)
            result_dict["f1_score"] = compute_f1_score(pred, gt)
            result_dict["normalized_prediction"] = normalize_text(pred)
            result_dict["normalized_ground_truth"] = normalize_text(gt)
        
        # Include expected sources from task nodes for comparison
        if task.get("nodes"):
            expected_sources = [
                node.get("source", "") for node in task["nodes"].values()
            ]
            result_dict["expected_sources"] = expected_sources
            
            # Compute source overlap
            if result.sources:
                pred_sources_set = set(str(s).lower() for s in result.sources)
                expected_sources_set = set(str(s).lower() for s in expected_sources if s)
                overlap = pred_sources_set & expected_sources_set
                result_dict["source_recall"] = (
                    len(overlap) / len(expected_sources_set)
                    if expected_sources_set else 0.0
                )
                result_dict["source_precision"] = (
                    len(overlap) / len(pred_sources_set)
                    if pred_sources_set else 0.0
                )
        
        # Include task complexity metadata
        result_dict["task_metadata"] = {
            "num_nodes": len(task.get("nodes", {})),
            "has_reasoning_chain": "reasoning_chain" in task,
        }
        
        logger.info(f"Completed task {task_index + 1}: {result_dict.get('success', False)}")
        return result_dict
        
    except Exception as e:
        logger.error(f"Error in task {task_index + 1}: {str(e)}")
        return {
            "task_id": task.get("id", task_index),
            "model": model,
            "question": task.get("question", ""),
            "ground_truth": task.get("answer", ""),
            "predicted_answer": "",
            "success": False,
            "error": str(e),
        }
    finally:
        # Clean up isolated sandbox
        _cleanup_isolated_sandbox(sandbox_path)


def _run_task_worker_process(
    result_queue: "multiprocessing.Queue",
    task: Dict[str, Any],
    task_index: int,
    model: str,
    config: "AgentConfig",
    log_dir: str,
    run_id: str,
    batch_name: Optional[str],
    sandbox_path_str: str,
) -> None:
    try:
        result = _run_task_worker(
            task,
            task_index,
            model,
            config,
            log_dir,
            run_id,
            batch_name,
            sandbox_path_str=sandbox_path_str,
        )
        result_queue.put((task_index, result, None))
    except BaseException as e:
        result_queue.put((task_index, None, f"{type(e).__name__}: {str(e)}"))


def _timeout_result(
    task: Dict[str, Any],
    task_index: int,
    model: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    return {
        "task_id": task.get("id", task_index),
        "model": model,
        "question": task.get("question", ""),
        "ground_truth": task.get("answer", ""),
        "predicted_answer": "",
        "reasoning": "",
        "sources_used": [],
        "datasets_used": [],
        "datasets_executed": [],
        "datasets_discovered": [],
        "tool_calls": 0,
        "tokens": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 0,
        "agent_cost_usd": 0.0,
        "cost": 0.0,
        "cost_source": "timeout",
        "cost_note": "Task process was killed at the configured wall-clock timeout.",
        "cost_breakdown": {},
        "time": float(timeout_seconds),
        "success": False,
        "error": f"Timeout: task exceeded {timeout_seconds} seconds and was killed",
    }


@dataclass
class AgentResult:
    """Result from an agent run."""
    answer: str
    reasoning: str = ""
    sources: List[str] = field(default_factory=list)
    datasets_used: List[str] = field(default_factory=list)
    datasets_executed: List[str] = field(default_factory=list)
    datasets_discovered: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_made: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    cost_source: str = ""
    cost_note: str = ""
    cost_breakdown: Dict[str, Any] = field(default_factory=dict)
    elapsed_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_turns: int = 25
    max_tokens: int = 8192
    temperature: float = 0.0
    system_prompt: Optional[str] = None
    verbose: bool = False
    timeout_seconds: int = 450  # 7.5 minutes default
    reasoning_effort: str = "medium"  # For OpenAI reasoning models: "low", "medium", "high"


class AgentRunner:
    """Runs an LLM agent with tool calling capabilities."""

    def __init__(
        self,
        model: str = "gpt-5.2",
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the agent runner.

        Args:
            model: Model name (e.g., "gpt-5.2", "bedrock/claude-opus-4.5")
            config: Agent configuration
            api_key: Optional API key (uses env var if not provided)
        """
        self.model = model
        self.config = config or AgentConfig()
        self.llm = LLMFactory.create(model, api_key, reasoning_effort=self.config.reasoning_effort)

        # Tool name -> function mapping
        self.tool_functions = {
            "search": agent_tools.search,
            "search_keyword": agent_tools.search_keyword,
            "list_files": agent_tools.list_files,
            "download": agent_tools.download,
            "get_sandbox_info": agent_tools.get_sandbox_info,
            "inspect_file": agent_tools.inspect_file,
            "execute_code": agent_tools.execute_code,
            "submit_answer": self._handle_submit_answer,
        }

        self._submitted_answer = None
        self._datasets_used = set()
        self._datasets_executed = set()
        self._datasets_discovered = set()

    def _execute_tool(self, tool_call: ToolCall) -> Any:
        """Execute a tool call and return the result."""
        if tool_call.name not in self.tool_functions:
            return {"error": f"Unknown tool: {tool_call.name}"}

        try:
            func = self.tool_functions[tool_call.name]
            result = func(**tool_call.arguments)
            return result
        except Exception as e:
            return {"error": str(e)}

    def _handle_submit_answer(
        self,
        answer: str,
        reasoning: str = "",
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Handle the submit_answer tool call."""
        cleaned_answer = _clean_answer(answer)
        self._submitted_answer = {
            "answer": cleaned_answer,
            "raw_answer": answer,
            "reasoning": reasoning,
            "sources": sources or [],
        }
        return {"status": "answer_submitted", "answer": cleaned_answer, "raw_answer": answer}

    def run(self, question: str) -> AgentResult:
        """
        Run the agent to answer a question using native tool calling.

        Args:
            question: The question to answer

        Returns:
            AgentResult with the answer and metadata
        """
        logger = get_logger()
        start_time = time.time()

        logger.info(f"=" * 60)
        logger.info(f"NEW TASK: {self.model}")
        logger.debug(f"QUESTION: {question}")
        logger.info(f"=" * 60)

        # Initialize state
        total_tokens = 0
        tool_calls_made = 0
        self._submitted_answer = None
        self._datasets_used = set()
        self._datasets_executed = set()
        self._datasets_discovered = set()

        try:
            return self._run_agent_loop(question, logger, start_time)
        finally:
            # Clean up sandbox after each task to prevent disk space issues
            try:
                agent_tools.cleanup_sandbox()
            except Exception as e:
                logger.warning(f"Failed to cleanup sandbox: {e}")

    def _run_agent_loop(self, question: str, logger, start_time: float) -> AgentResult:
        """Internal method that runs the main agent loop."""
        total_tokens = 0
        input_tokens = 0
        cached_input_tokens = 0
        cache_write_input_tokens = 0
        output_tokens = 0
        tool_calls_made = 0

        # Keep this task-independent so OpenAI prefix caching can reuse the
        # stable instruction/tool prefix across benchmark tasks.
        system_prompt = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

        # Conversation messages
        messages = [
            {
                "role": "user",
                "content": (
                    "QUESTION TO ANSWER:\n"
                    f"{question}\n\n"
                    "Please answer the question using the available tools. "
                    "Start by searching for relevant datasets. Use the available "
                    "tools to find and analyze data, then call submit_answer with "
                    "your final answer."
                ),
            }
        ]

        for turn in range(self.config.max_turns):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout_seconds:
                logger.warning(f"TIMEOUT after {elapsed:.1f}s")
                cost_info = LLMFactory.calculate_cost_breakdown(
                    self.model,
                    input_tokens,
                    output_tokens,
                    cached_input_tokens=cached_input_tokens,
                    cache_write_input_tokens=cache_write_input_tokens,
                )
                return AgentResult(
                    answer="",
                    datasets_used=sorted(self._datasets_used),
                    datasets_executed=sorted(self._datasets_executed),
                    datasets_discovered=sorted(self._datasets_discovered),
                    messages=messages,
                    tool_calls_made=tool_calls_made,
                    total_tokens=total_tokens,
                    input_tokens=input_tokens,
                    cached_input_tokens=cached_input_tokens,
                    cache_write_input_tokens=cache_write_input_tokens,
                    output_tokens=output_tokens,
                    total_cost=cost_info["cost_usd"],
                    cost_source=cost_info["source"],
                    cost_note=cost_info["note"],
                    cost_breakdown=cost_info,
                    elapsed_time=elapsed,
                    success=False,
                    error=f"Timeout: task exceeded {self.config.timeout_seconds} seconds",
                )

            logger.info(f"--- Turn {turn + 1} (elapsed: {elapsed:.1f}s) ---")

            try:
                # Call LLM with tools
                response = self.llm.complete_with_tools(
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    system=system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    tool_choice="auto" if turn < self.config.max_turns - 5 else "required",
                )

                input_tokens += response.usage.get("input_tokens", 0)
                cached_input_tokens += response.usage.get("cached_input_tokens", 0)
                cache_write_input_tokens += response.usage.get("cache_write_input_tokens", 0)
                output_tokens += response.usage.get("output_tokens", 0)
                total_tokens = input_tokens + output_tokens

                # Log response
                if response.content:
                    logger.debug(f"LLM content: {response.content}")
                logger.debug(f"Tool calls: {len(response.tool_calls)}")

                # If no native tool calls, try to parse from text content (fallback for some models)
                tool_call_parsed_from_text = False
                if not response.tool_calls and response.content:
                    parsed_tool_call = _parse_tool_call_from_text(response.content)
                    if parsed_tool_call:
                        logger.info(f"Parsed tool call from text: {parsed_tool_call.name}")
                        response.tool_calls.append(parsed_tool_call)
                        tool_call_parsed_from_text = True

                # If still no tool calls, prompt model to use tools
                if not response.tool_calls:
                    if response.content:
                        logger.warning(f"No tool calls, content: {response.content}")
                    # Add assistant message and prompt for tool use
                    messages.append({"role": "assistant", "content": response.content or ""})
                    messages.append({"role": "user", "content": "Please use one of the available tools to continue. If you have the answer, use submit_answer."})
                    continue

                # Process only ONE tool call per turn (take the first one)
                tool_call = response.tool_calls[0]
                # When tool call was parsed from text, don't include content (Bedrock Converse API constraint)
                assistant_content = "" if tool_call_parsed_from_text else (response.content or "")
                assistant_msg = {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_call.name, "arguments": json.dumps(tool_call.arguments)}
                        }
                    ]
                }
                messages.append(assistant_msg)

                tool_name = tool_call.name
                arguments = tool_call.arguments

                logger.info(f"Executing: {tool_name}({json.dumps(arguments)})")
                tool_calls_made += 1

                # Track dataset usage
                if tool_name == "list_files":
                    # list_files accepts dataset_ids (list)
                    ds_ids = arguments.get("dataset_ids", [])
                    if isinstance(ds_ids, list):
                        for ds_id in ds_ids:
                            normalized = _normalize_dataset_path(ds_id)
                            if normalized:
                                self._datasets_used.add(normalized)
                elif tool_name == "download":
                    # download accepts files (list of {dataset_id, file_path})
                    files = arguments.get("files", [])
                    if isinstance(files, list):
                        for file_spec in files:
                            if isinstance(file_spec, dict):
                                ds_id = _normalize_dataset_path(file_spec.get("dataset_id", ""))
                                if ds_id:
                                    self._datasets_used.add(ds_id)
                elif tool_name == "inspect_file":
                    dataset_id = _normalize_dataset_path(arguments.get("dataset_id", ""))
                    if dataset_id:
                        self._datasets_used.add(dataset_id)

                # Execute tool
                result = self._execute_tool(tool_call)

                # Track discovered datasets
                if tool_name in ("search", "search_keyword") and isinstance(result, dict):
                    for item in result.get("results", []):
                        ds_id = item.get("dataset_id", "")
                        if ds_id:
                            self._datasets_discovered.add(ds_id)

                # Track executed datasets
                if tool_name == "execute_code" and isinstance(result, dict):
                    code = arguments.get("code", "")
                    for ds_id in result.get("datasets_in_sandbox", []):
                        if ds_id and ds_id in code:
                            self._datasets_executed.add(ds_id)

                # Add tool result to conversation
                result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                # Truncate very long results to reduce context window usage
                if len(result_str) > 4000:
                    result_str = result_str[:2500] + "\n...[truncated]...\n" + result_str[-1000:]

                # Add remaining turns and time info
                remaining_turns = self.config.max_turns - turn - 1
                remaining_time = max(0, self.config.timeout_seconds - int(time.time() - start_time))
                status_info = f"\n\n[STATUS: {remaining_turns} turns remaining, {remaining_time}s time remaining]"
                result_str += status_info

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

                logger.debug(f"Tool result: {result_str}")

                # Check if answer was submitted
                if self._submitted_answer:
                    logger.info(f"ANSWER SUBMITTED: {self._submitted_answer.get('answer', '')}")
                    cost_info = LLMFactory.calculate_cost_breakdown(
                        self.model,
                        input_tokens,
                        output_tokens,
                        cached_input_tokens=cached_input_tokens,
                        cache_write_input_tokens=cache_write_input_tokens,
                    )
                    return AgentResult(
                        answer=self._submitted_answer.get("answer", ""),
                        reasoning=self._submitted_answer.get("reasoning", ""),
                        sources=self._submitted_answer.get("sources", []),
                        datasets_used=sorted(self._datasets_used),
                        datasets_executed=sorted(self._datasets_executed),
                        datasets_discovered=sorted(self._datasets_discovered),
                        messages=messages,
                        tool_calls_made=tool_calls_made,
                        total_tokens=total_tokens,
                        input_tokens=input_tokens,
                        cached_input_tokens=cached_input_tokens,
                        cache_write_input_tokens=cache_write_input_tokens,
                        output_tokens=output_tokens,
                        total_cost=cost_info["cost_usd"],
                        cost_source=cost_info["source"],
                        cost_note=cost_info["note"],
                        cost_breakdown=cost_info,
                        elapsed_time=time.time() - start_time,
                        success=True,
                    )

            except Exception as e:
                logger.error(f"Error in turn {turn + 1}: {e}")
                # Add error context and continue
                messages.append({
                    "role": "user",
                    "content": f"Error occurred: {str(e)[:500]}. Please try a different approach."
                })
                continue

        # Max turns reached
        logger.warning(f"MAX TURNS ({self.config.max_turns}) reached without answer")
        cost_info = LLMFactory.calculate_cost_breakdown(
            self.model,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
        )
        return AgentResult(
            answer="",
            datasets_used=sorted(self._datasets_used),
            datasets_executed=sorted(self._datasets_executed),
            datasets_discovered=sorted(self._datasets_discovered),
            messages=messages,
            tool_calls_made=tool_calls_made,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            output_tokens=output_tokens,
            total_cost=cost_info["cost_usd"],
            cost_source=cost_info["source"],
            cost_note=cost_info["note"],
            cost_breakdown=cost_info,
            elapsed_time=time.time() - start_time,
            success=False,
            error="Max turns reached without submitting an answer",
        )


# =============================================================================
# Batch Runner for Evaluation
# =============================================================================

class BatchRunner:
    """Run agent on multiple tasks for evaluation with parallel processing."""

    def __init__(
        self,
        model: str = "gpt-5.2",
        config: Optional[AgentConfig] = None,
        max_workers: Optional[int] = None,
    ):
        self.model = model
        self.config = config or AgentConfig()
        # If not specified, use number of CPUs (but cap at 4 for resource management)
        self.max_workers = max_workers or min(6, os.cpu_count() or 1)

    def run_tasks(
        self,
        tasks: List[Dict[str, Any]],
        verbose: bool = False,
        max_workers: Optional[int] = None,
        batch_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run agent on a list of tasks in parallel.

        Args:
            tasks: List of task dicts with 'question' and optionally 'answer' (ground truth)
            verbose: Print progress
            max_workers: Override number of workers for this run

        Returns:
            List of result dicts
        """
        num_workers = max_workers or self.max_workers
        
        # Create shared log directory and run id for this batch
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_label = batch_name or "batch"
        
        if verbose:
            print(f"\nRunning {len(tasks)} tasks in parallel with {num_workers} workers...")

        results = []
        active: Dict[int, Dict[str, Any]] = {}
        result_queue = multiprocessing.Queue()
        next_task_index = 0
        completed = 0

        def launch_task(task_index: int) -> None:
            sandbox_path = _create_isolated_sandbox(str(task_index))
            process = multiprocessing.Process(
                target=_run_task_worker_process,
                args=(
                    result_queue,
                    tasks[task_index],
                    task_index,
                    self.model,
                    self.config,
                    log_dir,
                    run_id,
                    batch_label,
                    str(sandbox_path),
                ),
            )
            process.start()
            active[task_index] = {
                "process": process,
                "start_time": time.time(),
                "sandbox_path": sandbox_path,
            }

        def record_result(task_index: int, result: Dict[str, Any]) -> None:
            nonlocal completed
            results.append((task_index, result))
            completed += 1
            if verbose:
                if result.get("success"):
                    print(f"✓ Task {task_index + 1}/{len(tasks)} completed")
                    if "exact_match" in result:
                        match_status = "✓" if result["exact_match"] == 1.0 else "✗"
                        print(f"  {match_status} Match: {result.get('exact_match', 0):.2f} | F1: {result.get('f1_score', 0):.3f}")
                else:
                    print(f"✗ Task {task_index + 1}/{len(tasks)} failed: {result.get('error', '')}")

        try:
            while completed < len(tasks):
                while next_task_index < len(tasks) and len(active) < num_workers:
                    launch_task(next_task_index)
                    next_task_index += 1

                while True:
                    try:
                        task_index, result, error = result_queue.get_nowait()
                    except queue.Empty:
                        break

                    info = active.pop(task_index, None)
                    if info is None:
                        continue
                    process = info["process"]
                    process.join(timeout=1)
                    _cleanup_isolated_sandbox(info["sandbox_path"])

                    if error:
                        result = {
                            "task_id": tasks[task_index].get("id", task_index),
                            "model": self.model,
                            "question": tasks[task_index].get("question", ""),
                            "ground_truth": tasks[task_index].get("answer", ""),
                            "predicted_answer": "",
                            "success": False,
                            "error": f"Worker error: {error}",
                        }
                    record_result(task_index, result)

                now = time.time()
                for task_index, info in list(active.items()):
                    if now - info["start_time"] < self.config.timeout_seconds:
                        continue

                    process = info["process"]
                    process.kill()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=1)
                    _cleanup_isolated_sandbox(info["sandbox_path"])
                    active.pop(task_index, None)
                    result = _timeout_result(
                        tasks[task_index],
                        task_index,
                        self.model,
                        self.config.timeout_seconds,
                    )
                    record_result(task_index, result)

                if completed < len(tasks):
                    time.sleep(0.2)
        finally:
            for task_index, info in list(active.items()):
                process = info["process"]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=3)
                _cleanup_isolated_sandbox(info["sandbox_path"])
            result_queue.close()
            result_queue.join_thread()

        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def run_from_files(
        self,
        task_files: List[str],
        verbose: bool = False,
        max_workers: Optional[int] = None,
        batch_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run agent on tasks loaded from JSON files with parallel processing.

        Args:
            task_files: List of paths to task JSON files
            verbose: Print progress
            max_workers: Override number of workers for this run

        Returns:
            List of result dicts
        """
        tasks = []
        for path in task_files:
            with open(path) as f:
                task = json.load(f)
                task["id"] = path
                tasks.append(task)

        # Derive batch name from task directory if not provided
        if not batch_name and task_files:
            try:
                common_dir = os.path.commonpath(task_files)
                batch_name = Path(common_dir).name
            except Exception:
                batch_name = None

        return self.run_tasks(tasks, verbose=verbose, max_workers=max_workers, batch_name=batch_name)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for running the agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM agent on data lake tasks")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--model", "-m", default="gpt-5.2", help="Model to use")
    parser.add_argument("--task-file", "-f", help="Path to task JSON file")
    parser.add_argument("--task-dir", "-d", help="Directory containing task JSON files")
    parser.add_argument("--max-turns", type=int, default=25, help="Max agent turns")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")

    args = parser.parse_args()

    config = AgentConfig(
        max_turns=args.max_turns,
        verbose=args.verbose,
    )

    # Single question mode
    if args.question:
        runner = AgentRunner(model=args.model, config=config)
        result = runner.run(args.question)

        print(f"\nAnswer: {result.answer}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")
        if result.sources:
            print(f"Sources: {result.sources}")
        cost_str = f"${result.total_cost:.4f}" if result.total_cost > 0 else "N/A"
        print(f"\nStats: {result.tool_calls_made} tool calls, {result.total_tokens} tokens, {result.elapsed_time:.2f}s, cost: {cost_str}")

        if not result.success:
            print(f"Error: {result.error}")

    # Task file mode
    elif args.task_file:
        batch = BatchRunner(model=args.model, config=config)
        results = batch.run_from_files([args.task_file], verbose=args.verbose)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    # Task directory mode
    elif args.task_dir:
        import glob
        task_files = sorted(glob.glob(f"{args.task_dir}/*.json"))
        print(f"Found {len(task_files)} task files")

        batch = BatchRunner(model=args.model, config=config)
        results = batch.run_from_files(task_files, verbose=args.verbose)

        # Compute summary statistics
        total = len(results)
        success = sum(1 for r in results if r["success"])
        exact_matches = sum(r.get("exact_match", 0) for r in results)
        f1_scores = [r.get("f1_score", 0) for r in results if "f1_score" in r]
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        avg_tool_calls = sum(r.get("tool_calls", 0) for r in results) / total if total else 0
        avg_tokens = sum(r.get("tokens", 0) for r in results) / total if total else 0
        avg_time = sum(r.get("time", 0) for r in results) / total if total else 0
        total_cost = sum(r.get("cost", 0) for r in results)
        avg_cost = total_cost / total if total else 0

        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - {args.model}")
        print(f"{'='*60}")
        print(f"Tasks:        {total}")
        print(f"Successful:   {success}/{total} ({100*success/total:.1f}%)")
        print(f"Exact Match:  {exact_matches}/{total} ({100*exact_matches/total:.1f}%)")
        print(f"Avg F1:       {avg_f1:.3f}")
        print(f"Avg Tools:    {avg_tool_calls:.1f} calls/task")
        print(f"Avg Tokens:   {avg_tokens:.0f} tokens/task")
        print(f"Avg Time:     {avg_time:.1f}s/task")
        print(f"Total Cost:   ${total_cost:.4f}")
        print(f"Avg Cost:     ${avg_cost:.4f}/task")
        print(f"{'='*60}")

        # Save results with summary
        output_data = {
            "model": args.model,
            "summary": {
                "total_tasks": total,
                "successful": success,
                "exact_match_count": exact_matches,
                "exact_match_rate": exact_matches / total if total else 0,
                "avg_f1_score": avg_f1,
                "avg_tool_calls": avg_tool_calls,
                "avg_tokens": avg_tokens,
                "avg_time_seconds": avg_time,
                "total_cost_usd": total_cost,
                "avg_cost_usd": avg_cost,
            },
            "results": results,
        }

        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
