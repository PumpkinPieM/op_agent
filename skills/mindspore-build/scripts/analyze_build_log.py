#!/usr/bin/env python3
"""
[EXPERIMENTAL] Build Log Analyzer for MindSpore.
Status: Not yet validated on real build logs. Use with caution.

Parses build output to extract errors, warnings, and suggested fixes.
Designed to be called by AI agents or developers for quick diagnosis.

Usage:
    python scripts/analyze_build_log.py /path/to/build.log
    python scripts/analyze_build_log.py /path/to/build.log --json
    python scripts/analyze_build_log.py /path/to/build.log --fix-suggestions

Exit codes:
    0   analysis complete, no errors found
    1   analysis complete, errors found
    2   file not found or unreadable
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict


@dataclass
class BuildIssue:
    file_path: str
    line_number: Optional[int]
    column: Optional[int]
    severity: str  # "error", "warning", "note"
    message: str
    source: str  # "gcc", "cmake", "ninja", "linker", "python"
    context: List[str]  # surrounding lines
    suggestion: Optional[str] = None


ERROR_PATTERNS = [
    # GCC/Clang compile errors
    {
        "pattern": r"^(.+?):(\d+):(\d+):\s+(error|warning|note):\s+(.+)$",
        "source": "gcc",
        "groups": {"file": 1, "line": 2, "col": 3, "severity": 4, "message": 5},
    },
    # CMake errors
    {
        "pattern": r"^CMake (Error|Warning)(?: at (.+?):(\d+))?\s*:\s*(.+)$",
        "source": "cmake",
        "groups": {"severity": 1, "file": 2, "line": 3, "message": 4},
    },
    # Ninja build errors
    {
        "pattern": r"^\[(\d+)/\d+\]\s+(FAILED|FAILED_WITH_MSG):\s+(.+)$",
        "source": "ninja",
        "groups": {"severity": 2, "message": 3},
    },
    # Linker errors
    {
        "pattern": r"^(.+?):(?:error|undefined reference):\s+(.+)$",
        "source": "linker",
        "groups": {"file": 1, "message": 2, "severity": "error"},
    },
    # Python import errors
    {
        "pattern": r"^(?:ImportError|ModuleNotFoundError):\s+(.+)$",
        "source": "python",
        "groups": {"message": 1, "severity": "error"},
    },
    # Generic error patterns
    {
        "pattern": r"^(?:FATAL|fatal|ERROR|Error):\s+(.+)$",
        "source": "generic",
        "groups": {"message": 1, "severity": "error"},
    },
]

SUGGESTIONS = {
    "cannot find -l": "Library not found. Check if the library is installed and linker paths are correct.",
    "undefined reference to": "Symbol not found. Check if all required libraries are linked and their versions match.",
    "fatal error: (.+\\.h): No such file": "Header file not found. Check if the development package is installed.",
    "CMake Error: Could not find (.+)": "CMake dependency not found. Install the package or set CMAKE_PREFIX_PATH.",
    "Permission denied": "Permission issue. Check file/directory permissions or run with appropriate privileges.",
    "out of memory": "OOM during build. Reduce -j thread count or increase available memory.",
    "Killed": "Process killed, likely OOM. Reduce -j thread count.",
    "No space left on device": "Disk full. Clean up build/ directory or free disk space.",
    "connection refused": "Network connection failed. Check if the server is running and accessible.",
    "SSL certificate": "SSL/TLS error. Check certificates or use --insecure for testing.",
    "submodule": "Git submodule issue. Run: git submodule update --init --recursive",
    "npu-smi: command not found": "NPU driver not installed or not in PATH.",
    "CANN.*not found": "CANN toolkit not installed or ASCEND_CUSTOM_PATH not set.",
    "CUDA.*not found": "CUDA toolkit not installed or PATH not set.",
    "Python.h: No such file": "Python development headers missing. Install python3-dev.",
    "numpy.*version": "NumPy version mismatch. Check requirements.",
    "recompile with -fPIC": "Position-independent code issue. Add -fPIC to compiler flags.",
}


def parse_log(log_content: str) -> List[BuildIssue]:
    """Parse build log and extract issues."""
    issues = []
    lines = log_content.splitlines()
    
    for i, line in enumerate(lines):
        for pattern_info in ERROR_PATTERNS:
            match = re.match(pattern_info["pattern"], line, re.IGNORECASE)
            if match:
                groups = pattern_info["groups"]
                issue = BuildIssue(
                    file_path=match.group(groups.get("file", 0)) if groups.get("file") else "",
                    line_number=int(match.group(groups["line"])) if groups.get("line") and match.group(groups["line"]) else None,
                    column=int(match.group(groups["col"])) if groups.get("col") and match.group(groups["col"]) else None,
                    severity=match.group(groups["severity"]).lower() if groups.get("severity") and isinstance(groups["severity"], int) 
                             else groups.get("severity", "error"),
                    message=match.group(groups["message"]) if groups.get("message") else "",
                    source=pattern_info["source"],
                    context=get_context(lines, i, 3),
                )
                issue.suggestion = find_suggestion(issue.message)
                issues.append(issue)
                break
    
    return issues


def get_context(lines: List[str], center_idx: int, radius: int) -> List[str]:
    """Get surrounding lines for context."""
    start = max(0, center_idx - radius)
    end = min(len(lines), center_idx + radius + 1)
    return lines[start:end]


def find_suggestion(message: str) -> Optional[str]:
    """Find a suggestion based on error message patterns."""
    message_lower = message.lower()
    for pattern, suggestion in SUGGESTIONS.items():
        if re.search(pattern.lower(), message_lower):
            return suggestion
    return None


def deduplicate_issues(issues: List[BuildIssue]) -> List[BuildIssue]:
    """Remove duplicate issues based on file, line, and message."""
    seen = set()
    unique = []
    for issue in issues:
        key = (issue.file_path, issue.line_number, issue.message[:100])
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique


def summarize_issues(issues: List[BuildIssue]) -> Dict:
    """Generate summary statistics."""
    by_severity = defaultdict(int)
    by_source = defaultdict(int)
    by_file = defaultdict(int)
    
    for issue in issues:
        by_severity[issue.severity] += 1
        by_source[issue.source] += 1
        if issue.file_path:
            by_file[issue.file_path] += 1
    
    return {
        "total": len(issues),
        "by_severity": dict(by_severity),
        "by_source": dict(by_source),
        "top_files": sorted(by_file.items(), key=lambda x: -x[1])[:10],
    }


def print_report(issues: List[BuildIssue], summary: Dict, show_suggestions: bool = True):
    """Print human-readable report."""
    print("=" * 70)
    print("BUILD LOG ANALYSIS REPORT")
    print("=" * 70)
    
    print(f"\nSummary:")
    print(f"  Total issues: {summary['total']}")
    print(f"  By severity: {summary['by_severity']}")
    print(f"  By source: {summary['by_source']}")
    
    if summary['top_files']:
        print(f"\nTop files with issues:")
        for file_path, count in summary['top_files'][:5]:
            print(f"  - {file_path}: {count} issues")
    
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        print(f"\n{'=' * 70}")
        print(f"ERRORS ({len(errors)} found)")
        print("=" * 70)
        for i, issue in enumerate(errors[:20], 1):
            print(f"\n[{i}] {issue.file_path}")
            if issue.line_number:
                print(f"    Line {issue.line_number}" + (f", Col {issue.column}" if issue.column else ""))
            print(f"    {issue.message}")
            if show_suggestions and issue.suggestion:
                print(f"    Suggestion: {issue.suggestion}")
        
        if len(errors) > 20:
            print(f"\n... and {len(errors) - 20} more errors")
    
    warnings = [i for i in issues if i.severity == "warning"]
    if warnings and len(warnings) <= 10:
        print(f"\n{'=' * 70}")
        print(f"WARNINGS ({len(warnings)} found)")
        print("=" * 70)
        for issue in warnings:
            print(f"  - {issue.file_path}:{issue.line_number or '?'}: {issue.message[:80]}")


def main():
    ap = argparse.ArgumentParser(description="Analyze MindSpore build log")
    ap.add_argument("log_file", help="Path to build log file")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    ap.add_argument("--fix-suggestions", action="store_true", default=True,
                    help="Include fix suggestions (default: True)")
    ap.add_argument("--max-errors", type=int, default=50,
                    help="Maximum number of errors to report (default: 50)")
    args = ap.parse_args()

    if not os.path.exists(args.log_file):
        print(f"FATAL: File not found: {args.log_file}", file=sys.stderr)
        return 2

    try:
        with open(args.log_file, "r", encoding="utf-8", errors="replace") as f:
            log_content = f.read()
    except Exception as e:
        print(f"FATAL: Cannot read file: {e}", file=sys.stderr)
        return 2

    issues = parse_log(log_content)
    issues = deduplicate_issues(issues)
    issues = issues[:args.max_errors]
    summary = summarize_issues(issues)

    if args.json:
        output = {
            "summary": summary,
            "issues": [asdict(i) for i in issues],
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(issues, summary, args.fix_suggestions)

    # Return 1 if errors found, 0 otherwise
    has_errors = any(i.severity == "error" for i in issues)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
