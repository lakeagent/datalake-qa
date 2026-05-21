"""
Tool Schema Definitions for LLM Agent

Defines tools for data lake access:
1. search - Find datasets/files by prefix
2. download - Download files to local sandbox
3. execute_code - Run Python code against downloaded data
4. search_keyword - Keyword search across Wikipedia and data.gov with S3 validation
"""

AGENT_TOOLS = [
    {
        "name": "inspect_file",
        "description": "Inspect a file and return safe metadata (type, size, delimiter guess, header, keys) without returning raw content.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "Dataset identifier (no folder prefix)"
                },
                "file_path": {
                    "type": "string",
                    "description": "File path relative to dataset root"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to inspect (default 5)",
                    "default": 5
                }
            },
            "required": ["dataset_id", "file_path"]
        }
    },
    {
        "name": "search",
        "description": "Search for datasets by name prefix. Returns matching dataset identifiers (no folder prefixes).",
        "parameters": {
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Dataset name prefix (at least 2 characters). Examples: 'census'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default 50)",
                    "default": 50
                }
            },
            "required": ["prefix"]
        }
    },
    {
        "name": "search_keyword",
        "description": "Tag-style keyword search. Returns dataset identifiers (no folder prefixes) sorted by relevance. Note: limit=k may miss relevant datasets because results are ranked by relevance.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Tag-style keyword to search"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return after merging and ranking"
                }
            },
            "required": ["keyword"]
        }
    },
    {
        "name": "list_files",
        "description": "List files within a dataset. Use the dataset identifier returned by search().",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "Dataset identifier returned by search() (no folder prefix)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum files to return (default 100)",
                    "default": 100
                }
            },
            "required": ["dataset_id"]
        }
    },
    {
        "name": "download",
        "description": "Download a file to the local sandbox directory. Use dataset_id plus a relative file path (from list_files).",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "Dataset identifier returned by search() (no folder prefix)"
                },
                "file_path": {
                    "type": "string",
                    "description": "File path relative to the dataset root (use paths returned by list_files)"
                }
            },
            "required": ["dataset_id", "file_path"]
        }
    },
    {
        "name": "get_sandbox_info",
        "description": "Get information about the current sandbox directory and list all downloaded files.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "execute_code",
        "description": """Execute Python code in a sandbox environment to analyze or preview downloaded files.

The code runs with:
- Working directory set to sandbox
- Pre-imported: pandas, json, csv, os, glob, re, Path
- Variable SANDBOX_DIR = path to sandbox directory
- Variable FILES = list of downloaded file paths

Use print() to output results. The printed output will be returned.
You can use this tool both to query data and to view/preview files (txt, csv, etc).
Note: execution has a timeout; avoid inefficient code.

Example:
    df = pd.read_csv(FILES[0])
    print(df.head())
    print(f"Rows: {len(df)}")""",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() for output."
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "submit_answer",
        "description": "Submit the final answer to the question. Use this when you have found the answer. The answer must be wrapped in square brackets like [Answer].",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the question"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of how you arrived at the answer"
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths used to find the answer"
                }
            },
            "required": ["answer"]
        }
    }
]


def get_tools_for_provider(provider: str = "openai") -> list:
    """
    Get tools formatted for a specific provider.

    Args:
        provider: "openai" or "bedrock" (Anthropic via AWS Bedrock)

    Returns:
        List of tool definitions in provider-specific format
    """
    if provider == "openai":
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                }
            }
            for tool in AGENT_TOOLS
        ]
    elif provider in ("anthropic", "bedrock"):
        # Bedrock uses Anthropic's format for Claude models
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in AGENT_TOOLS
        ]
    else:
        return AGENT_TOOLS  # Return unified format


# Export
__all__ = ["AGENT_TOOLS", "get_tools_for_provider"]
