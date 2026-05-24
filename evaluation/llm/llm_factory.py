"""
LLM Factory - Unified interface for OpenAI and AWS Bedrock models.

Supported models:
- gpt-5.2, gpt-5-mini (OpenAI)
- bedrock/claude-opus-4.5, bedrock/claude-haiku-4.5 (AWS Bedrock)

Usage:
    from evaluation.llm.llm_factory import LLMFactory

    # Create client
    llm = LLMFactory.create("gpt-5.2")

    # Simple completion
    response = llm.complete("What is 2+2?")

    # With tools
    response = llm.complete_with_tools(
        messages=[{"role": "user", "content": "Search for climate data"}],
        tools=tools_schema
    )
"""

import os
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
from enum import Enum

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return None

load_dotenv()


def _get_attr_or_item(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _openai_usage_dict(usage: Any) -> Dict[str, int]:
    prompt_details = _get_attr_or_item(usage, "prompt_tokens_details", None)
    cached_tokens = _get_attr_or_item(prompt_details, "cached_tokens", 0) or 0
    return {
        "input_tokens": _get_attr_or_item(usage, "prompt_tokens", 0) or 0,
        "cached_input_tokens": cached_tokens,
        "output_tokens": _get_attr_or_item(usage, "completion_tokens", 0) or 0,
    }


def _bedrock_usage_dict(usage: Dict[str, Any]) -> Dict[str, int]:
    return {
        "input_tokens": usage.get("inputTokens", usage.get("input_tokens", 0)) or 0,
        "cached_input_tokens": usage.get(
            "cacheReadInputTokensCount",
            usage.get("cache_read_input_tokens", 0),
        ) or 0,
        "output_tokens": usage.get("outputTokens", usage.get("output_tokens", 0)) or 0,
        "cache_write_input_tokens": usage.get(
            "cacheWriteInputTokensCount",
            usage.get("cache_creation_input_tokens", 0),
        ) or 0,
    }


def _extract_json_metadata(content: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    text = content or ""

    # Handle empty response
    if not text.strip():
        metadata["json_valid"] = False
        metadata["json_error"] = "empty response from LLM"
        metadata["raw_json"] = ""
        return metadata

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = text.strip()

    # Handle empty after stripping
    if not candidate:
        metadata["json_valid"] = False
        metadata["json_error"] = "empty content after stripping markdown"
        metadata["raw_json"] = text[:500]
        return metadata

    try:
        parsed = json.loads(candidate)
        metadata["parsed_json"] = parsed
        metadata["json_valid"] = True
        return metadata
    except json.JSONDecodeError as e:
        metadata["json_valid"] = False
        metadata["json_error"] = str(e)
        metadata["raw_json"] = candidate[:500]
        return metadata


class Provider(Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    BEDROCK_CONVERSE = "bedrock_converse"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"  # Prompt-based tool calling for DeepSeek models


@dataclass
class ToolCall:
    """Represents a tool/function call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Unified response format across all providers."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Simple text completion."""
        pass

    @abstractmethod
    def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Completion with tool/function calling."""
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        expect_json: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate response from messages with optional JSON expectation."""
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """Streaming text completion."""
        pass


# =============================================================================
# OpenAI Implementation
# =============================================================================

class OpenAILLM(BaseLLM):
    """OpenAI GPT models"""

    MODELS = {
        "gpt-5.2": "gpt-5.2",
        "gpt-5-mini": "gpt-5-mini",
    }

    # Models that don't support custom temperature (only default=1.0)
    NO_TEMPERATURE_MODELS = {"gpt-5-mini", "gpt-5.2"}

    # Models that support reasoning_effort parameter
    REASONING_MODELS = {"gpt-5.2"}

    def __init__(self, model: str, api_key: Optional[str] = None, reasoning_effort: str = "medium"):
        super().__init__(model, api_key or os.getenv("OPENAI_API_KEY"))
        self._supports_temperature = model not in self.NO_TEMPERATURE_MODELS
        self._supports_reasoning = model in self.REASONING_MODELS
        self._reasoning_effort = reasoning_effort  # "low", "medium", or "high"
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if self._supports_temperature:
            kwargs["temperature"] = temperature
        if self._supports_reasoning:
            kwargs["reasoning_effort"] = self._reasoning_effort

        response = self.client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            finish_reason=response.choices[0].finish_reason,
            model=response.model,
            usage=_openai_usage_dict(response.usage),
            raw_response=response,
        )

    def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        # Prepend system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages

        # Convert tools to OpenAI format
        openai_tools = self._convert_tools_to_openai(tools)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": openai_tools if openai_tools else None,
            "tool_choice": tool_choice if openai_tools else None,
            "max_completion_tokens": max_tokens,
            "parallel_tool_calls": False,
        }
        if self._supports_temperature:
            kwargs["temperature"] = temperature
        if self._supports_reasoning:
            kwargs["reasoning_effort"] = self._reasoning_effort

        response = self.client.chat.completions.create(**kwargs)

        # Parse tool calls
        tool_calls = []
        msg = response.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
            model=response.model,
            usage=_openai_usage_dict(response.usage),
            raw_response=response,
            metadata={},
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        expect_json: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        response_format = {"type": "json_object"} if expect_json else None
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "response_format": response_format,
        }
        if self._supports_temperature:
            kwargs["temperature"] = temperature
        if self._supports_reasoning:
            kwargs["reasoning_effort"] = self._reasoning_effort

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content = msg.content or ""
        llm_response = LLMResponse(
            content=content,
            finish_reason=response.choices[0].finish_reason,
            model=response.model,
            usage=_openai_usage_dict(response.usage),
            raw_response=response,
            metadata={},
        )
        if expect_json:
            llm_response.metadata.update(_extract_json_metadata(content))
        return llm_response

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
        if self._supports_temperature:
            kwargs["temperature"] = temperature

        stream = self.client.chat.completions.create(**kwargs)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _convert_tools_to_openai(self, tools: List[Dict]) -> List[Dict]:
        """Convert unified tool format to OpenAI format."""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            })
        return openai_tools


# =============================================================================
# OpenAI-Compatible API Implementation (Together AI, Fireworks, OpenRouter, etc.)
# =============================================================================

class OpenAICompatibleLLM(BaseLLM):
    """OpenAI-compatible API providers (Together AI, Fireworks, OpenRouter, vLLM, etc.)"""

    # Provider configurations: provider_name -> (base_url, env_var_for_api_key)
    PROVIDERS = {
        "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
        "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    }

    # Model aliases for convenience
    MODEL_ALIASES = {
        "together/qwen3-235b": "Qwen/Qwen3-235B-A22B",
        "together/qwen3-32b": "Qwen/Qwen3-32B",
        "together/llama4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "together/deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "fireworks/qwen3-235b": "accounts/fireworks/models/qwen3-235b-a22b",
        "fireworks/llama4-maverick": "accounts/fireworks/models/llama4-maverick-instruct-basic",
        "openrouter/qwen3-235b": "qwen/qwen3-235b-a22b",
    }

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model, api_key)

        # Parse provider from model string (e.g., "together/qwen3-235b")
        if "/" in model:
            self.provider = model.split("/")[0]
        else:
            self.provider = "together"  # default

        # Get provider config
        if self.provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {self.provider}. Available: {list(self.PROVIDERS.keys())}")

        base_url, api_key_env = self.PROVIDERS[self.provider]
        self.api_key = api_key or os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(f"API key required. Set {api_key_env} environment variable.")

        # Resolve model alias to actual model ID
        self.actual_model = self.MODEL_ALIASES.get(model, model.split("/", 1)[-1] if "/" in model else model)

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.actual_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            finish_reason=response.choices[0].finish_reason,
            model=response.model,
            usage=_openai_usage_dict(response.usage),
            raw_response=response,
        )

    def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        if system:
            messages = [{"role": "system", "content": system}] + messages

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            }
            for tool in tools
        ]

        kwargs = {
            "model": self.actual_model,
            "messages": messages,
            "tools": openai_tools if openai_tools else None,
            "tool_choice": tool_choice if openai_tools else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = self.client.chat.completions.create(**kwargs)

        tool_calls = []
        msg = response.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
            model=response.model,
            usage=_openai_usage_dict(response.usage),
            raw_response=response,
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        expect_json: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs = {
            "model": self.actual_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if expect_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""

        llm_response = LLMResponse(
            content=content,
            finish_reason=response.choices[0].finish_reason,
            model=response.model,
            usage=_openai_usage_dict(response.usage),
            raw_response=response,
        )
        if expect_json:
            llm_response.metadata.update(_extract_json_metadata(content))
        return llm_response

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.actual_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# =============================================================================
# AWS Bedrock Implementation
# =============================================================================

class BedrockLLM(BaseLLM):
    """AWS Bedrock models (Claude via Bedrock)"""

    # Using cross-region inference (us. prefix) for broader access
    MODELS = {
        "bedrock/claude-opus-4.5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "bedrock/claude-haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    }

    def __init__(self, model: str, api_key: Optional[str] = None, region: str = "us-east-1"):
        super().__init__(model, api_key)
        self.region = region
        try:
            import boto3
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=region,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            )
        except ImportError:
            raise ImportError("boto3 package required: pip install boto3")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system

        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
        )

        result = json.loads(response['body'].read())

        content = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        return LLMResponse(
            content=content,
            finish_reason=result.get("stop_reason", "stop"),
            model=self.model,
            usage=_bedrock_usage_dict(result.get("usage", {})),
            raw_response=result,
        )

    def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        # Convert messages to Bedrock/Anthropic format
        bedrock_messages = self._convert_messages(messages)

        # Convert tools to Anthropic format
        bedrock_tools = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
            }
            for tool in tools
        ]

        # Convert tool_choice
        if tool_choice == "auto":
            tc = {"type": "auto"}
        elif tool_choice == "none":
            tc = {"type": "none"}
        elif tool_choice == "required":
            tc = {"type": "any"}
        else:
            tc = {"type": "tool", "name": tool_choice}

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": bedrock_messages,
            "tools": bedrock_tools,
            "tool_choice": tc,
        }
        if system:
            body["system"] = system

        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
        )

        result = json.loads(response['body'].read())

        # Parse response
        content = ""
        tool_calls = []

        for block in result.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=result.get("stop_reason", "stop"),
            model=self.model,
            usage=_bedrock_usage_dict(result.get("usage", {})),
            raw_response=result,
            metadata={},
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        expect_json: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        system = None
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
                break
        bedrock_messages = self._convert_messages(messages)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": bedrock_messages,
        }
        if system:
            body["system"] = system

        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
        )

        result = json.loads(response['body'].read())

        content = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        llm_response = LLMResponse(
            content=content,
            finish_reason=result.get("stop_reason", "stop"),
            model=self.model,
            usage=_bedrock_usage_dict(result.get("usage", {})),
            raw_response=result,
            metadata={},
        )
        if expect_json:
            llm_response.metadata.update(_extract_json_metadata(content))
        return llm_response

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        messages = [{"role": "user", "content": prompt}]

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system

        response = self.client.invoke_model_with_response_stream(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
        )

        for event in response.get('body', []):
            chunk = json.loads(event['chunk']['bytes'])
            if chunk.get('type') == 'content_block_delta':
                delta = chunk.get('delta', {})
                if delta.get('type') == 'text_delta':
                    yield delta.get('text', '')

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style messages to Bedrock/Anthropic format."""
        converted = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue  # System is handled separately

            if role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }]
                })
            elif role == "assistant" and msg.get("tool_calls"):
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    # Get arguments - may be a JSON string (OpenAI format) or dict
                    args = tc.get("function", {}).get("arguments", tc.get("arguments", {}))
                    # Parse JSON string to dict if needed (Bedrock requires dict)
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    content.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", tc.get("name", "")),
                        "input": args,
                    })
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append({"role": role, "content": msg.get("content", "")})

        return converted


# =============================================================================
# AWS Bedrock Converse API (for Llama, Mistral, etc.)
# =============================================================================

class BedrockConverseLLM(BaseLLM):
    """AWS Bedrock models using Converse API (Llama, Mistral, etc.)"""

    def __init__(self, model: str, api_key: Optional[str] = None, region: str = "us-east-1"):
        super().__init__(model, api_key)
        self.region = region
        try:
            import boto3
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=region,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            )
        except ImportError:
            raise ImportError("boto3 package required: pip install boto3")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        kwargs = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = self.client.converse(**kwargs)

        content = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                content += block["text"]

        usage = response.get("usage", {})
        return LLMResponse(
            content=content,
            finish_reason=response.get("stopReason", "stop"),
            model=self.model,
            usage=_bedrock_usage_dict(usage),
            raw_response=response,
        )

    def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        # Convert messages to Converse format
        converse_messages = self._convert_messages(messages)

        # Convert tools to Converse format
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "inputSchema": {
                            "json": tool.get("parameters", {"type": "object", "properties": {}})
                        },
                    }
                }
                for tool in tools
            ]
        }

        # Set tool choice
        if tool_choice == "auto":
            tool_config["toolChoice"] = {"auto": {}}
        elif tool_choice == "required":
            tool_config["toolChoice"] = {"any": {}}
        elif tool_choice != "none":
            tool_config["toolChoice"] = {"tool": {"name": tool_choice}}

        kwargs = {
            "modelId": self.model,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
            "toolConfig": tool_config,
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = self.client.converse(**kwargs)

        # Parse response
        content = ""
        tool_calls = []
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                content += block["text"]
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(ToolCall(
                    id=tool_use.get("toolUseId", ""),
                    name=tool_use.get("name", ""),
                    arguments=tool_use.get("input", {}),
                ))

        usage = response.get("usage", {})
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.get("stopReason", "stop"),
            model=self.model,
            usage=_bedrock_usage_dict(usage),
            raw_response=response,
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        expect_json: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        converse_messages = self._convert_messages(messages)

        # Extract system from messages if present
        system = None
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
                break

        kwargs = {
            "modelId": self.model,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = self.client.converse(**kwargs)

        content = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                content += block["text"]

        usage = response.get("usage", {})
        llm_response = LLMResponse(
            content=content,
            finish_reason=response.get("stopReason", "stop"),
            model=self.model,
            usage=_bedrock_usage_dict(usage),
            raw_response=response,
        )
        if expect_json:
            llm_response.metadata.update(_extract_json_metadata(content))
        return llm_response

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        kwargs = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = self.client.converse_stream(**kwargs)

        for event in response.get("stream", []):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    yield delta["text"]

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style messages to Converse format."""
        converted = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue  # System is handled separately

            if role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": msg.get("tool_call_id", ""),
                            "content": [{"text": msg.get("content", "")}],
                        }
                    }]
                })
            elif role == "assistant" and msg.get("tool_calls"):
                # Bedrock Converse API doesn't allow mixing text and toolUse in same content array
                # Only include toolUse blocks
                content = []
                for tc in msg["tool_calls"]:
                    args = tc.get("function", {}).get("arguments", tc.get("arguments", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    content.append({
                        "toolUse": {
                            "toolUseId": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", tc.get("name", "")),
                            "input": args,
                        }
                    })
                converted.append({"role": "assistant", "content": content})
            else:
                content_text = msg.get("content", "")
                converted.append({"role": role, "content": [{"text": content_text}]})

        return converted


# =============================================================================
# DeepSeek Implementation (Prompt-based Tool Calling)
# =============================================================================

class DeepSeekLLM(BedrockConverseLLM):
    """
    DeepSeek models via Bedrock with prompt-based tool calling.
    
    DeepSeek R1 doesn't support native tool calling, so we use a specialized
    prompt that instructs the model to output JSON tool calls which we parse.
    """

    # Prompt template for tool calling
    TOOL_CALLING_PROMPT = """You have access to the following tools:

{tool_definitions}

## IMPORTANT INSTRUCTIONS
1. Think through the problem step by step inside <thinking> tags
2. When you need to use a tool, output ONLY a JSON object (no markdown, no explanation outside thinking):
   {{"name": "tool_name", "parameters": {{"param1": "value1"}}}}
3. When you have the final answer and want to submit it, use:
   {{"name": "make_decision", "parameters": {{"answer": "your answer here"}}}}
4. Output exactly ONE tool call per response, then STOP
5. Do NOT simulate tool results - wait for the actual result

Available tools:
{tool_list}"""

    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> tuple:
        """Format tools into text descriptions for the prompt."""
        tool_definitions = []
        tool_names = []
        
        for tool in tools:
            name = tool["name"]
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            
            tool_names.append(name)
            
            # Format parameters
            props = params.get("properties", {})
            required = params.get("required", [])
            
            param_strs = []
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req = "(required)" if pname in required else "(optional)"
                param_strs.append(f"    - {pname} ({ptype}) {req}: {pdesc}")
            
            params_text = "\n".join(param_strs) if param_strs else "    (no parameters)"
            
            tool_definitions.append(f"### {name}\n{desc}\nParameters:\n{params_text}")
        
        return "\n\n".join(tool_definitions), ", ".join(tool_names)

    def _parse_tool_call_from_text(self, content: str) -> Optional[ToolCall]:
        """Parse a tool call from the model's text output."""
        import time
        
        if not content:
            return None

        text = content.strip()
        
        # Remove <thinking>...</thinking> blocks to get the actual output
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()
        
        # Remove markdown code fences if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Try to find JSON object in remaining text
        # Look for the last JSON object (in case there's multiple)
        json_matches = list(re.finditer(r'\{[^{}]*\}', text, re.DOTALL))
        
        for match in reversed(json_matches):
            try:
                data = json.loads(match.group())
                if isinstance(data, dict) and "name" in data:
                    name = data.get("name")
                    params = data.get("parameters", data.get("arguments", {}))
                    if isinstance(params, str):
                        try:
                            params = json.loads(params)
                        except json.JSONDecodeError:
                            params = {}
                    
                    return ToolCall(
                        id=f"deepseek_{int(time.time()*1000)}",
                        name=name,
                        arguments=params if isinstance(params, dict) else {}
                    )
            except json.JSONDecodeError:
                continue
        
        # Try parsing the entire remaining text as JSON
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "name" in data:
                name = data.get("name")
                params = data.get("parameters", data.get("arguments", {}))
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError:
                        params = {}
                
                return ToolCall(
                    id=f"deepseek_{int(time.time()*1000)}",
                    name=name,
                    arguments=params if isinstance(params, dict) else {}
                )
        except json.JSONDecodeError:
            pass
        
        return None

    def _convert_messages_for_prompt_tools(self, messages: List[Dict]) -> List[Dict]:
        """
        Convert messages to Converse format, but handle tool calls/results as text.
        This is needed because we're doing prompt-based tool calling, not native.
        """
        converted = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue  # System is handled separately

            if role == "tool":
                # Convert tool result to a user message with the result
                tool_result = msg.get("content", "")
                converted.append({
                    "role": "user",
                    "content": [{"text": f"Tool result:\n{tool_result}"}]
                })
            elif role == "assistant" and msg.get("tool_calls"):
                # Convert assistant tool calls to text representation
                tc = msg["tool_calls"][0] if msg["tool_calls"] else None
                if tc:
                    args = tc.get("function", {}).get("arguments", tc.get("arguments", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    name = tc.get("function", {}).get("name", tc.get("name", ""))
                    tool_json = json.dumps({"name": name, "parameters": args})
                    
                    # Include any thinking/content from the original message
                    text_content = msg.get("content", "")
                    if text_content:
                        full_content = f"{text_content}\n{tool_json}"
                    else:
                        full_content = tool_json
                    
                    converted.append({
                        "role": "assistant",
                        "content": [{"text": full_content}]
                    })
            else:
                content_text = msg.get("content", "")
                converted.append({"role": role, "content": [{"text": content_text}]})

        return converted

    def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """
        Complete with tools using prompt-based approach.
        
        Instead of using native tool calling (which DeepSeek doesn't support),
        we inject tool definitions into the system prompt and parse JSON from output.
        """
        # Format tools into prompt text
        tool_definitions, tool_list = self._format_tools_for_prompt(tools)
        
        # Build enhanced system prompt
        tool_prompt = self.TOOL_CALLING_PROMPT.format(
            tool_definitions=tool_definitions,
            tool_list=tool_list
        )
        
        if system:
            enhanced_system = f"{system}\n\n{tool_prompt}"
        else:
            enhanced_system = tool_prompt

        # Convert messages (handling tool calls/results as text)
        converse_messages = self._convert_messages_for_prompt_tools(messages)

        kwargs = {
            "modelId": self.model,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
            "system": [{"text": enhanced_system}],
        }

        response = self.client.converse(**kwargs)

        # Extract text content
        content = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                content += block["text"]

        # Parse tool call from text output
        tool_calls = []
        parsed_tool = self._parse_tool_call_from_text(content)
        if parsed_tool:
            tool_calls.append(parsed_tool)

        usage = response.get("usage", {})
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.get("stopReason", "stop"),
            model=self.model,
            usage=_bedrock_usage_dict(usage),
            raw_response=response,
            metadata={"prompt_based_tools": True},
        )


# =============================================================================
# LLM Factory
# =============================================================================

class LLMFactory:
    """Factory for creating LLM instances."""

    # Model pricing: cost per 1M tokens (input, output)
    # Update these values based on current provider pricing
    MODEL_PRICING = {
        # OpenAI (example pricing - update with actual values)
        "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
        "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},

        # AWS Bedrock Claude models
        "bedrock/claude-opus-4.5": {"input": 5.00, "output": 25.00},
        "bedrock/claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
        "bedrock/claude-haiku-4.5": {"input": 1.00, "output": 5.00},

        # AWS Bedrock Llama models
        "bedrock/llama4-maverick": {"input": 0.24, "output": 0.97},
        "bedrock/llama4-scout": {"input": 0.17, "output": 0.66},
        "bedrock/llama3.3-70b": {"input": 0.72, "output": 0.72},
        "bedrock/llama3.1-70b": {"input": 0.72, "output": 0.72},

        # AWS Bedrock Mistral models
        "bedrock/mistral-large": {"input": 0.50, "output": 1.50},
        "bedrock/mixtral-8x7b": {"input": 0.45, "output": 0.70},

        # AWS Bedrock Qwen models
        "bedrock/qwen3-32b": {"input": 0.15, "output": 0.60},
        "bedrock/qwen3-80b": {"input": 0.15, "output": 1.20},
        "bedrock/qwen3-235b": {"input": 0.53, "output": 2.66},

        # AWS Bedrock DeepSeek
        "bedrock/deepseek-r1": {"input": 1.35, "output": 5.40},

        # Together AI
        "together/qwen3-235b": {"input": 0.65, "output": 3.00},
        "together/qwen3-32b": {"input": 0.50, "output": 1.50},
        "together/llama4-maverick": {"input": 0.27, "output": 0.85},
        "together/deepseek-r1": {"input": 3.00, "output": 7.00},

        # Fireworks AI
        "fireworks/qwen3-235b": {"input": 0.22, "output": 0.88},
        "fireworks/llama4-maverick": {"input": 0.17, "output": 0.17},

        # OpenRouter
        "openrouter/qwen3-235b": {"input": 1.00, "output": 1.00},
    }

    # Model name -> (Provider, actual_model_id)
    # Evaluation targets: GPT-5.2, GPT-5-mini, Claude Opus 4.5, Claude Haiku 4.5
    MODEL_REGISTRY = {
        # OpenAI
        "gpt-5.2": (Provider.OPENAI, "gpt-5.2"),
        "gpt-5-mini": (Provider.OPENAI, "gpt-5-mini"),

        # AWS Bedrock (Claude models) - using cross-region inference (us. prefix)
        "bedrock/claude-opus-4.5": (Provider.BEDROCK, "us.anthropic.claude-opus-4-5-20251101-v1:0"),
        "bedrock/claude-sonnet-4.5": (Provider.BEDROCK, "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        "bedrock/claude-haiku-4.5": (Provider.BEDROCK, "us.anthropic.claude-haiku-4-5-20251001-v1:0"),

        # AWS Bedrock (Llama models) - using Converse API with cross-region inference (us. prefix)
        "bedrock/llama4-maverick": (Provider.BEDROCK_CONVERSE, "us.meta.llama4-maverick-17b-instruct-v1:0"),
        "bedrock/llama4-scout": (Provider.BEDROCK_CONVERSE, "us.meta.llama4-scout-17b-instruct-v1:0"),
        "bedrock/llama3.3-70b": (Provider.BEDROCK_CONVERSE, "us.meta.llama3-3-70b-instruct-v1:0"),
        "bedrock/llama3.1-70b": (Provider.BEDROCK_CONVERSE, "us.meta.llama3-1-70b-instruct-v1:0"),

        # AWS Bedrock (Mistral models) - using Converse API
        "bedrock/mistral-large": (Provider.BEDROCK_CONVERSE, "mistral.mistral-large-3-675b-instruct"),
        "bedrock/mixtral-8x7b": (Provider.BEDROCK_CONVERSE, "mistral.mixtral-8x7b-instruct-v0:1"),

        # AWS Bedrock (Qwen models) - using Converse API
        "bedrock/qwen3-32b": (Provider.BEDROCK_CONVERSE, "qwen.qwen3-32b-v1:0"),
        "bedrock/qwen3-80b": (Provider.BEDROCK_CONVERSE, "qwen.qwen3-next-80b-a3b"),
        "bedrock/qwen3-235b": (Provider.BEDROCK_CONVERSE, "qwen.qwen3-vl-235b-a22b"),

        # AWS Bedrock (DeepSeek models) - using prompt-based tool calling
        "bedrock/deepseek-r1": (Provider.DEEPSEEK, "us.deepseek.r1-v1:0"),

        # Together AI
        "together/qwen3-235b": (Provider.TOGETHER, "together/qwen3-235b"),
        "together/qwen3-32b": (Provider.TOGETHER, "together/qwen3-32b"),
        "together/llama4-maverick": (Provider.TOGETHER, "together/llama4-maverick"),
        "together/deepseek-r1": (Provider.TOGETHER, "together/deepseek-r1"),

        # Fireworks AI
        "fireworks/qwen3-235b": (Provider.FIREWORKS, "fireworks/qwen3-235b"),
        "fireworks/llama4-maverick": (Provider.FIREWORKS, "fireworks/llama4-maverick"),

        # OpenRouter
        "openrouter/qwen3-235b": (Provider.OPENROUTER, "openrouter/qwen3-235b"),
    }

    @classmethod
    def create(
        cls,
        model: str,
        api_key: Optional[str] = None,
        reasoning_effort: str = "medium",
    ) -> BaseLLM:
        """
        Create an LLM instance.

        Args:
            model: Model name (e.g., "gpt-5.2", "bedrock/claude-opus-4.5")
            api_key: Optional API key (uses env var if not provided)
            reasoning_effort: For OpenAI reasoning models - "low", "medium", or "high"

        Returns:
            LLM instance

        Example:
            >>> llm = LLMFactory.create("gpt-5.2", reasoning_effort="high")
            >>> response = llm.complete("Hello!")
        """
        if model not in cls.MODEL_REGISTRY:
            # Try to infer provider from model name
            if model.startswith("gpt-"):
                provider = Provider.OPENAI
                actual_model = model
            elif model.startswith("bedrock/"):
                provider = Provider.BEDROCK
                actual_model = model
            elif model.startswith("together/"):
                provider = Provider.TOGETHER
                actual_model = model
            elif model.startswith("fireworks/"):
                provider = Provider.FIREWORKS
                actual_model = model
            elif model.startswith("openrouter/"):
                provider = Provider.OPENROUTER
                actual_model = model
            else:
                raise ValueError(
                    f"Unknown model: {model}. Available: {list(cls.MODEL_REGISTRY.keys())}"
                )
        else:
            provider, actual_model = cls.MODEL_REGISTRY[model]

        if provider == Provider.OPENAI:
            return OpenAILLM(actual_model, api_key, reasoning_effort=reasoning_effort)
        elif provider == Provider.BEDROCK:
            return BedrockLLM(actual_model, api_key)
        elif provider == Provider.BEDROCK_CONVERSE:
            return BedrockConverseLLM(actual_model, api_key)
        elif provider == Provider.DEEPSEEK:
            return DeepSeekLLM(actual_model, api_key)
        elif provider in (Provider.TOGETHER, Provider.FIREWORKS, Provider.OPENROUTER):
            return OpenAICompatibleLLM(actual_model, api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def list_models(cls) -> Dict[str, List[str]]:
        """List all available models grouped by provider."""
        models = {p.value: [] for p in Provider}
        for name, (provider, _) in cls.MODEL_REGISTRY.items():
            models[provider.value].append(name)
        return models

    @classmethod
    def calculate_cost(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_write_input_tokens: int = 0,
    ) -> float:
        """
        Calculate the cost for a given number of tokens.

        Args:
            model: Model name (e.g., "gpt-5.2", "bedrock/claude-opus-4.5")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        return cls.calculate_cost_breakdown(
            model,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
        )["cost_usd"]

    @classmethod
    def calculate_cost_breakdown(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_write_input_tokens: int = 0,
    ) -> Dict[str, Any]:
        pricing = cls.MODEL_PRICING.get(model)
        if not pricing:
            return {
                "cost_usd": 0.0,
                "source": "missing_model_pricing",
                "note": "No configured per-token pricing for this model.",
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "cache_write_input_tokens": cache_write_input_tokens,
                "uncached_input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_rate_per_1m": None,
                "cached_input_rate_per_1m": None,
                "cache_write_input_rate_per_1m": None,
                "output_rate_per_1m": None,
            }

        # Pricing is per 1M tokens
        cached_tokens = min(max(cached_input_tokens, 0), max(input_tokens, 0))
        cache_write_tokens = min(max(cache_write_input_tokens, 0), max(input_tokens - cached_tokens, 0))
        uncached_input_tokens = max(input_tokens - cached_tokens - cache_write_tokens, 0)
        cached_input_rate = pricing.get("cached_input", pricing["input"])
        cache_write_input_rate = pricing.get("cache_write_input", pricing["input"])
        input_cost = (uncached_input_tokens / 1_000_000) * pricing["input"]
        cached_input_cost = (cached_tokens / 1_000_000) * cached_input_rate
        cache_write_input_cost = (cache_write_tokens / 1_000_000) * cache_write_input_rate
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return {
            "cost_usd": input_cost + cached_input_cost + cache_write_input_cost + output_cost,
            "source": "configured_token_rates",
            "note": (
                "Per-task provider billing cost is not returned by model APIs; "
                "this is computed from reported token usage and configured rates."
            ),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "cache_write_input_tokens": cache_write_tokens,
            "uncached_input_tokens": uncached_input_tokens,
            "output_tokens": output_tokens,
            "input_rate_per_1m": pricing["input"],
            "cached_input_rate_per_1m": cached_input_rate,
            "cache_write_input_rate_per_1m": cache_write_input_rate,
            "output_rate_per_1m": pricing["output"],
        }

    @classmethod
    def get_pricing(cls, model: str) -> Optional[Dict[str, float]]:
        """Get pricing info for a model (per 1M tokens)."""
        return cls.MODEL_PRICING.get(model)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_llm(model: str, api_key: Optional[str] = None) -> BaseLLM:
    """Shortcut for LLMFactory.create()"""
    return LLMFactory.create(model, api_key)


def complete(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """One-shot completion - returns just the text."""
    llm = get_llm(model)
    response = llm.complete(prompt, system, temperature, max_tokens)
    return response.content


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "LLMFactory",
    "BaseLLM",
    "OpenAILLM",
    "OpenAICompatibleLLM",
    "BedrockLLM",
    "BedrockConverseLLM",
    "DeepSeekLLM",
    "LLMResponse",
    "ToolCall",
    "Provider",
    "get_llm",
    "complete",
]
