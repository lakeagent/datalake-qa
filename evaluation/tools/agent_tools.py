"""
Data Lake Access Tools for LLM Agent

This module provides tools for LLM agents to access the data lake:

1. search(prefix) - Find datasets matching a prefix (searches both wikipedia and datagov)
2. download(dataset_id, file_path) - Download a file to local sandbox
3. execute_code(code) - Execute Python code against downloaded data
4. search_keyword(query) - Keyword search across wikipedia and data.gov with S3 validation

Bucket: lakeqa-yc4103-datalake
Folders: wikipedia/, datagov/
"""

import os
import re
import sys
import tempfile
import shutil
import traceback
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, Any, List

import boto3
import requests
from dotenv import load_dotenv

# Load AWS credentials from .env
load_dotenv()

# Configuration
BUCKET = "lakeqa-yc4103-datalake"
FOLDERS = ["wikipedia", "datagov"]
REGION = "us-east-1"

# Sandbox directory on main disk (500G) instead of /tmp (63G tmpfs)
SANDBOX_BASE_DIR = Path(__file__).resolve().parent.parent.parent / ".sandbox"

# Global sandbox directory (created per session)
_SANDBOX_DIR = None
# Optional override to force a specific sandbox directory (set by callers for isolation)
_SANDBOX_OVERRIDE = None


def set_sandbox_dir(path: Path) -> None:
    """Force the sandbox directory to a specific path (per-process isolation)."""
    global _SANDBOX_DIR, _SANDBOX_OVERRIDE
    _SANDBOX_OVERRIDE = Path(path)
    _SANDBOX_OVERRIDE.mkdir(parents=True, exist_ok=True)
    _SANDBOX_DIR = _SANDBOX_OVERRIDE


def _get_s3_client():
    """Get authenticated S3 client."""
    return boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )


def _get_sandbox_dir() -> Path:
    """Get or create the sandbox directory for downloaded files."""
    global _SANDBOX_DIR, _SANDBOX_OVERRIDE

    # If a caller pinned the sandbox (per-process isolation), use it
    if _SANDBOX_OVERRIDE is not None:
        _SANDBOX_DIR = _SANDBOX_OVERRIDE
        return _SANDBOX_DIR

    if _SANDBOX_DIR is None or not _SANDBOX_DIR.exists():
        SANDBOX_BASE_DIR.mkdir(parents=True, exist_ok=True)
        _SANDBOX_DIR = Path(tempfile.mkdtemp(prefix="task_", dir=SANDBOX_BASE_DIR))
    return _SANDBOX_DIR


def _dataset_exists(s3, folder: str, dataset_id: str) -> bool:
    """Check whether a dataset exists under a given folder."""
    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=f"{folder}/{dataset_id}/",
        MaxKeys=1
    )
    return "Contents" in response or "CommonPrefixes" in response


def _resolve_dataset_folder(dataset_id: str) -> Optional[str]:
    """Resolve dataset folder (datagov or wikipedia) for a dataset_id."""
    if not dataset_id:
        return None
    s3 = _get_s3_client()
    matches = []
    for folder in FOLDERS:
        if _dataset_exists(s3, folder, dataset_id):
            matches.append(folder)
    if len(matches) == 1:
        return matches[0]
    return None


def _guess_delimiter(line: str) -> str:
    candidates = [",", "\t", "|", ";"]
    best = ""
    best_count = 0
    for c in candidates:
        count = line.count(c)
        if count > best_count:
            best_count = count
            best = c
    return best if best_count > 0 else ""


def _looks_like_json(text: str) -> bool:
    t = text.strip()
    return t.startswith("{") or t.startswith("[")


def inspect_file(dataset_id: str, file_path: str, max_lines: int = 5) -> Dict[str, Any]:
    """
    Inspect a file and return safe metadata (no raw content).
    """
    if not dataset_id:
        return {'error': "dataset_id is required"}
    if not file_path:
        return {'error': "file_path is required"}

    folder = _resolve_dataset_folder(dataset_id)
    if folder is None:
        return {'error': f"Dataset not found or ambiguous: {dataset_id}"}

    s3 = _get_s3_client()
    key = f"{folder}/{dataset_id}/{file_path.lstrip('/')}"

    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
        size = head.get("ContentLength", 0)
    except Exception as e:
        return {'error': f"Failed to stat file: {str(e)}"}

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key, Range="bytes=0-65535")
        raw = obj["Body"].read()
        text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        return {'error': f"Failed to read file: {str(e)}"}

    lines = text.splitlines()
    preview = lines[: max_lines or 5]
    first_line = preview[0] if preview else ""

    meta = {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "size_bytes": size,
        "line_count_sampled": len(preview),
        "looks_like_json": _looks_like_json(first_line),
    }

    if first_line:
        meta["delimiter_guess"] = _guess_delimiter(first_line)
        if meta["delimiter_guess"]:
            meta["header_columns"] = [c.strip() for c in first_line.split(meta["delimiter_guess"])]

    # If JSON, attempt to parse the first line for keys (safe)
    if meta["looks_like_json"]:
        try:
            first_obj = json.loads(first_line)
            if isinstance(first_obj, dict):
                meta["json_keys"] = sorted(first_obj.keys())
        except Exception:
            pass

    return meta


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text) if text else ""
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_by_query(query_tokens: List[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0
    query_set = set(query_tokens)
    common = query_set.intersection(text_tokens)
    if not common:
        return 0.0
    coverage = len(common) / len(query_set)
    density = len(common) / len(text_tokens)
    return (coverage * 0.8) + (density * 0.2)


def _search_wikipedia_titles(query: str) -> List[Dict[str, Any]]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
    }
    headers = {"User-Agent": "DataLakeAgentTools/1.0"}

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    results = data.get("query", {}).get("search", [])
    return [{"title": item.get("title"), "api_score": item.get("score")} for item in results]


def _search_datagov_packages(query: str) -> List[Dict[str, Any]]:
    url = "https://catalog.data.gov/api/3/action/package_search"
    params = {"q": query}
    headers = {"User-Agent": "DataLakeAgentTools/1.0"}

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    if not data.get("success"):
        raise RuntimeError("data.gov search failed")
    return data.get("result", {}).get("results", [])


# =============================================================================
# Tool 1: Search
# =============================================================================

def search(prefixes: List[str], limit: int = 50) -> Dict[str, Any]:
    """
    Search for datasets matching one or more prefixes.

    Searches across BOTH wikipedia/ and datagov/ folders automatically.
    Uses S3's native prefix search which is efficient even with billions of objects.

    Args:
        prefixes: List of search prefixes. Each prefix must be at least 2 characters.
                  Examples: ["Barack"], ["climate", "census", "weather"]
        limit: Maximum results per folder per prefix (default 50)

    Returns:
        Dict with 'results' containing dataset identifiers

    Examples:
        >>> search(["Barack"])
        >>> search(["Barack", "climate", "census"])
    """
    # Validate input
    if not isinstance(prefixes, list):
        return {'error': "prefixes must be a list of strings."}
    prefix_list = prefixes

    # Validate all prefixes
    for p in prefix_list:
        if not p or len(p) < 2:
            return {'error': f"Prefix '{p}' must be at least 2 characters."}

    results_by_prefix = {}
    all_results = []
    seen_ids = set()

    for prefix in prefix_list:
        prefix_results = []
        for folder in FOLDERS:
            s3 = _get_s3_client()
            full_prefix = f"{folder}/{prefix}"

            # First try to find datasets (directories)
            response = s3.list_objects_v2(
                Bucket=BUCKET,
                Prefix=full_prefix,
                Delimiter='/',
                MaxKeys=limit
            )

            # Get dataset-level results (CommonPrefixes are "directories")
            if 'CommonPrefixes' in response:
                for p in response['CommonPrefixes']:
                    dataset_id = p['Prefix'].split('/')[1]
                    result_entry = {'dataset_id': dataset_id, 'type': 'dataset'}
                    prefix_results.append(result_entry)
                    if dataset_id not in seen_ids:
                        seen_ids.add(dataset_id)
                        all_results.append(result_entry)

        results_by_prefix[prefix] = prefix_results

    return {
        'results': all_results,
        'results_by_prefix': results_by_prefix,
        'count': len(all_results),
        'prefixes': prefix_list
    }


def search_keyword(
    keywords: List[str],
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tag-style keyword search filtered by S3 existence.

    Args:
        keywords: List of short tag-style keywords. Long sentences will impair performance.
                  Examples: ["police"], ["police", "crime", "traffic"]
        limit: Optional cap on results after ranking; may omit relevant datasets

    Returns:
        Dict with 'results' list of dataset identifiers and metadata.
    """
    # Validate input
    if not isinstance(keywords, list):
        return {'error': "keywords must be a list of strings."}
    keyword_list = keywords

    # Validate all keywords
    for kw in keyword_list:
        if not kw or not kw.strip():
            return {'error': "All keywords must be non-empty."}

    s3 = _get_s3_client()
    results = []
    seen_ids = set()
    results_by_keyword = {}

    for keyword in keyword_list:
        query_tokens = _tokenize(keyword)
        keyword_results = []

        try:
            wiki_hits = _search_wikipedia_titles(keyword)
        except Exception:
            wiki_hits = []

        for item in wiki_hits:
            title = item.get("title") or ""
            dataset_id = title.replace(' ', '_')
            if dataset_id and _dataset_exists(s3, "wikipedia", dataset_id):
                result_entry = {
                    "title": title,
                    "dataset_id": dataset_id,
                    "score": _score_by_query(query_tokens, title),
                }
                keyword_results.append(result_entry)
                if dataset_id not in seen_ids:
                    seen_ids.add(dataset_id)
                    results.append(result_entry)

        try:
            datagov_hits = _search_datagov_packages(keyword)
        except Exception:
            datagov_hits = []

        for item in datagov_hits:
            name = item.get("name") or ""
            title = item.get("title") or name
            if name and _dataset_exists(s3, "datagov", name):
                score_text = f"{title} {name}".strip()
                result_entry = {
                    "title": title,
                    "dataset_id": name,
                    "score": _score_by_query(query_tokens, score_text),
                }
                keyword_results.append(result_entry)
                if name not in seen_ids:
                    seen_ids.add(name)
                    results.append(result_entry)

        # Sort and clean keyword-specific results
        keyword_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        results_by_keyword[keyword] = [{"title": r.get("title", ""), "dataset_id": r.get("dataset_id", "")} for r in keyword_results]

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    if limit is not None and limit < len(results):
        results = results[:limit]

    cleaned = [{"title": r.get("title", ""), "dataset_id": r.get("dataset_id", "")} for r in results]

    return {
        "results": cleaned,
        "results_by_keyword": results_by_keyword,
        "count": len(results),
        "keywords": keyword_list
    }


def list_files(dataset_ids: List[str], limit: int = 100) -> Dict[str, Any]:
    """
    List files within one or more datasets/directories.

    WARNING: Only use this for datasets with a SMALL number of files (< 100).
    The data lake contains billions of objects. If you try to list files in
    a large dataset or use a broad path, this operation may be very slow or
    return truncated results. Always provide a specific dataset path.

    Args:
        dataset_ids: List of dataset identifiers.
                     Examples: ["Barack_Obama"], ["Barack_Obama", "climate-data"]
        limit: Maximum files to return per dataset (default 100)

    Returns:
        Dict with 'files' list grouped by dataset_id

    Example:
        >>> list_files(["Barack_Obama"])
        >>> list_files(["Barack_Obama", "climate-data"])
    """
    # Validate input
    if not isinstance(dataset_ids, list):
        return {'error': "dataset_ids must be a list of strings."}
    id_list = dataset_ids

    # Validate all dataset_ids
    for ds_id in id_list:
        if not ds_id:
            return {'error': "All dataset_ids must be non-empty."}

    s3 = _get_s3_client()
    all_files = []
    results_by_dataset = {}
    any_truncated = False

    for dataset_id in id_list:
        folder = _resolve_dataset_folder(dataset_id)
        if folder is None:
            results_by_dataset[dataset_id] = {'error': f"Dataset not found or ambiguous: {dataset_id}"}
            continue

        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=f"{folder}/{dataset_id}/",
            MaxKeys=limit
        )

        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if not obj['Key'].endswith('/'):
                    relative_path = obj['Key'].split(f"{folder}/{dataset_id}/", 1)[-1]
                    file_entry = {
                        'path': relative_path,
                        'size': obj['Size'],
                        'dataset_id': dataset_id
                    }
                    files.append(file_entry)
                    all_files.append(file_entry)

        results_by_dataset[dataset_id] = {
            'files': files,
            'count': len(files),
            'truncated': response.get('IsTruncated', False)
        }
        if response.get('IsTruncated', False):
            any_truncated = True

    return {
        'files': all_files,
        'count': len(all_files),
        'dataset_ids': id_list,
        'by_dataset': results_by_dataset,
        'truncated': any_truncated
    }


# =============================================================================
# Tool 2: Download
# =============================================================================

def download(files: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Download one or more files from S3 to the local sandbox directory.

    Args:
        files: List of file specifications, each with 'dataset_id' and 'file_path'.
               Maximum 5 files per call.
               Example: [{"dataset_id": "Barack_Obama", "file_path": "table_0.csv"}]

    Returns:
        Dict with 'downloaded' list of successful downloads and 'sandbox_dir'

    Example:
        >>> download([{"dataset_id": "Barack_Obama", "file_path": "table_0.csv"}])
        >>> download([
        ...     {"dataset_id": "Barack_Obama", "file_path": "content.txt"},
        ...     {"dataset_id": "climate-data", "file_path": "data.csv"}
        ... ])
    """
    if not isinstance(files, list):
        return {'error': "files must be a list of {dataset_id, file_path} objects"}

    if len(files) > 5:
        return {'error': "Maximum 5 files per download call"}

    if len(files) == 0:
        return {'error': "files list cannot be empty"}

    s3 = _get_s3_client()
    sandbox = _get_sandbox_dir()

    downloaded = []
    errors = []

    for file_spec in files:
        if not isinstance(file_spec, dict):
            errors.append({'error': "Each file must be a dict with dataset_id and file_path"})
            continue

        dataset_id = file_spec.get('dataset_id', '')
        file_path = file_spec.get('file_path', '')

        if not dataset_id:
            errors.append({'error': "dataset_id is required", 'file_spec': file_spec})
            continue
        if not file_path:
            errors.append({'error': "file_path is required", 'dataset_id': dataset_id})
            continue

        folder = _resolve_dataset_folder(dataset_id)
        if folder is None:
            errors.append({'error': f"Dataset not found: {dataset_id}", 'dataset_id': dataset_id, 'file_path': file_path})
            continue

        s3_key = f"{folder}/{dataset_id}/{file_path.lstrip('/')}"

        # Create local path structure (no folder prefix)
        local_path = sandbox / dataset_id / file_path.lstrip('/')
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3.download_file(BUCKET, s3_key, str(local_path))
            file_size = local_path.stat().st_size

            downloaded.append({
                'local_path': str(local_path),
                'file_path': file_path,
                'dataset_id': dataset_id,
                'size': file_size,
                'status': 'downloaded'
            })
        except Exception as e:
            errors.append({
                'error': f"Failed to download: {str(e)}",
                'dataset_id': dataset_id,
                'file_path': file_path
            })

    result = {
        'downloaded': downloaded,
        'download_count': len(downloaded),
        'sandbox_dir': str(sandbox)
    }

    if errors:
        result['errors'] = errors

    return result


def get_sandbox_info() -> Dict[str, Any]:
    """
    Get information about the current sandbox directory and downloaded files.

    Returns:
        Dict with sandbox_dir path and list of downloaded files
    """
    sandbox = _get_sandbox_dir()

    files = []
    total_size = 0
    for path in sandbox.rglob('*'):
        if path.is_file():
            size = path.stat().st_size
            files.append({
                'path': str(path),
                'relative_path': str(path.relative_to(sandbox)),
                'size': size
            })
            total_size += size

    return {
        'sandbox_dir': str(sandbox),
        'files': files,
        'file_count': len(files),
        'total_size': total_size
    }


# =============================================================================
# Tool 3: Execute Code (Python Sandbox)
# =============================================================================

def execute_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a sandbox environment with access to downloaded files.

    The code runs with:
    - Working directory set to the sandbox directory
    - Pre-imported: pandas, json, csv, os, glob, re
    - Variable `SANDBOX_DIR` pointing to the sandbox directory
    - Variable `FILES` containing list of downloaded file paths

    Write your analysis code and print() results. The printed output will be returned.
    You can use this tool both to query data and to view/preview files (txt, csv, etc).
    Note: execution has a timeout; avoid inefficient code.

    Args:
        code: Python code to execute

    Returns:
        Dict with 'output' (stdout), 'error' (if any), 'success' (bool)

    Example:
        >>> execute_code('''
        ... import pandas as pd
        ... df = pd.read_csv(SANDBOX_DIR + "/wikipedia/Barack_Obama/table_0.csv")
        ... print(df.head())
        ... print(f"Total rows: {len(df)}")
        ... ''')
        {'output': '   col1  col2\\n...\\nTotal rows: 50', 'success': True}
    """
    if not code or not code.strip():
        return {'error': "No code provided", 'success': False}

    sandbox = _get_sandbox_dir()

    # Collect downloaded files
    downloaded_files = []
    for path in sandbox.rglob('*'):
        if path.is_file():
            downloaded_files.append(str(path))
    sandbox_files = []
    datasets_in_sandbox = set()
    for file_path in downloaded_files:
        try:
            rel_path = Path(file_path).relative_to(sandbox)
        except ValueError:
            continue
        sandbox_files.append(str(rel_path))
        if rel_path.parts:
            datasets_in_sandbox.add(rel_path.parts[0])

    # Prepare execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        'SANDBOX_DIR': str(sandbox),
        'FILES': downloaded_files,
    }

    # Block all outgoing network traffic by disabling socket
    import socket as _socket
    _original_socket = _socket.socket

    def _blocked_socket(*args, **kwargs):
        raise OSError("Network access is disabled in sandbox. Use the download() tool to fetch data from the datalake.")

    _socket.socket = _blocked_socket

    # Pre-import common libraries
    pre_imports = """
import pandas as pd
import json
import csv
import os
import glob
import re
from pathlib import Path
"""

    # Capture stdout
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_cwd = os.getcwd()

    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        # Change to sandbox directory
        os.chdir(sandbox)

        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Execute pre-imports
        exec(pre_imports, exec_globals)

        # Execute user code
        exec(code, exec_globals)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        result = {
            'output': output,
            'success': True,
            'sandbox_dir': str(sandbox),
            'sandbox_files': sandbox_files,
            'datasets_in_sandbox': sorted(datasets_in_sandbox)
        }

        if errors:
            result['stderr'] = errors

        return result

    except BaseException as e:
        # Catch BaseException to handle SystemExit, KeyboardInterrupt, etc.
        # that the agent's code might raise (these bypass "except Exception")
        return {
            'output': stdout_capture.getvalue(),
            'error': f"{type(e).__name__}: {str(e)}",
            'traceback': traceback.format_exc(),
            'success': False,
            'sandbox_dir': str(sandbox),
            'sandbox_files': sandbox_files,
            'datasets_in_sandbox': sorted(datasets_in_sandbox)
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.chdir(old_cwd)
        _socket.socket = _original_socket  # Restore network access


# =============================================================================
# Utility Functions
# =============================================================================

def cleanup_sandbox() -> Dict[str, Any]:
    """
    Clean up the sandbox directory and delete all downloaded files.

    Returns:
        Dict with cleanup status
    """
    global _SANDBOX_DIR

    if _SANDBOX_DIR is None or not _SANDBOX_DIR.exists():
        return {'status': 'no_sandbox', 'deleted_files': 0}

    try:
        file_count = sum(1 for _ in _SANDBOX_DIR.rglob('*') if _.is_file())
        shutil.rmtree(_SANDBOX_DIR)
        _SANDBOX_DIR = None
        _SANDBOX_OVERRIDE = None
        return {'status': 'cleaned', 'deleted_files': file_count}
    except Exception as e:
        return {'error': f"Failed to cleanup: {str(e)}"}


# Export all public functions
__all__ = [
    'search',
    'list_files',
    'download',
    'inspect_file',
    'get_sandbox_info',
    'execute_code',
    'cleanup_sandbox',
    'set_sandbox_dir',
]
