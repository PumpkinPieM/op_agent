from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from common import DEFAULT_OUT_DIR, REPO_ROOT, write_csv, write_jsonl


DEFAULT_OP_PLUGIN_ROOT = REPO_ROOT / "op-plugin"
DEFAULT_FUNCTIONS_YAML = DEFAULT_OP_PLUGIN_ROOT / "op_plugin" / "config" / "op_plugin_functions.yaml"
DEFAULT_DERIVATIVES_YAML = DEFAULT_OP_PLUGIN_ROOT / "op_plugin" / "config" / "derivatives.yaml"
DEFAULT_OPAPI_ROOT = DEFAULT_OP_PLUGIN_ROOT / "op_plugin" / "ops" / "opapi"
DEFAULT_OPAPI_HELPER_ROOT = DEFAULT_OP_PLUGIN_ROOT / "op_plugin" / "utils" / "custom_functions" / "opapi"

EXEC_CMD_RE = re.compile(r"\bEXEC_NPU(?:_NO_FORMAT_CHECK)?_CMD(?:_SYNC)?\s*\(\s*(aclnn[A-Z][A-Za-z0-9_]*)")
GEN_EXEC_RE = re.compile(r"^aclnn[A-Z][A-Za-z0-9_]*$")
FUNCTION_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
FUNC_NAME_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\.]*)\s*\((.*)\)\s*->\s*(.+)$")
DELEGATE_CALL_RE = re.compile(
    r"\b(?:op_api|at_npu::native::custom_ops)::([A-Za-z_][A-Za-z0-9_]*)\s*\("
)

HELPER_PREPROCESS_PATTERNS = (
    "value_or(",
    ".item(",
    "copy_scalar_to_device",
    "is_scalar_wrapped_to_tensor",
    "const_cast<",
    ".t()",
    "static_cast<",
    "reinterpret_cast<",
    "sym_sizes()",
    "std::vector<",
)

CUSTOM_OUTPUT_PATTERNS = (
    "at::empty(",
    "std::make_tuple(",
    "std::tie(",
    "output_size =",
    "result0",
    "result1",
    "result2",
    "result3",
    "apply_tensor(",
)

FUNCTION_NAME_BLACKLIST = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "static_cast",
    "reinterpret_cast",
    "const_cast",
    "dynamic_cast",
    "TORCH_CHECK",
    "DO_COMPATIBILITY",
    "EXEC_NPU_CMD",
    "EXEC_NPU_NO_FORMAT_CHECK_CMD",
    "EXEC_NPU_CMD_SYNC",
}


def short_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def normalize_root_name(name: str) -> str:
    value = (name or "").strip()
    if not value:
        return ""
    value = re.sub(r"\.(Tensor|Scalar|out|Tensor_mode|Scalar_mode|out_mode)$", "", value)
    if "." in value:
        value = value.split(".", 1)[0]
    if value.endswith("_") and not value.endswith("__"):
        value = value[:-1]
    return value


def normalize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).lower()


def split_top_level(text: str, sep: str = ",") -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth_round = 0
    depth_square = 0
    depth_angle = 0
    in_single = False
    in_double = False
    for char in text:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if char == "(":
                depth_round += 1
            elif char == ")":
                depth_round = max(0, depth_round - 1)
            elif char == "[":
                depth_square += 1
            elif char == "]":
                depth_square = max(0, depth_square - 1)
            elif char == "<":
                depth_angle += 1
            elif char == ">":
                depth_angle = max(0, depth_angle - 1)
            elif char == sep and depth_round == 0 and depth_square == 0 and depth_angle == 0:
                token = "".join(current).strip()
                if token:
                    parts.append(token)
                current = []
                continue
        current.append(char)
    token = "".join(current).strip()
    if token:
        parts.append(token)
    return parts


def parse_default_value(value: str):
    token = value.strip()
    if not token:
        return None
    if token == "None":
        return None
    if token in {"true", "false"}:
        return token == "true"
    if token in {"True", "False"}:
        return token == "True"
    try:
        return ast.literal_eval(token)
    except Exception:
        return token


def parse_param(token: str) -> Optional[dict]:
    token = token.strip()
    if not token or token == "*":
        return None
    if "=" in token:
        left, default = token.split("=", 1)
        required = False
        default_value = parse_default_value(default)
    else:
        left = token
        required = True
        default_value = None
    left = left.strip()
    if " " not in left:
        return None
    type_name, name = left.rsplit(" ", 1)
    return {
        "name": name.strip(),
        "type": type_name.strip(),
        "required": required,
        "default": default_value,
    }


def parse_returns(token: str) -> List[dict]:
    text = token.strip()
    if text == "()":
        return []
    if text.startswith("(") and text.endswith(")"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        items = split_top_level(inner)
        return [{"name": f"output{i}", "type": item.strip()} for i, item in enumerate(items)]
    return [{"name": "output", "type": text}]


def parse_func_decl(func_decl: str) -> Tuple[str, str, List[dict], List[dict]]:
    match = FUNC_NAME_RE.match(func_decl.strip())
    if not match:
        raise ValueError(f"Unsupported PTA function declaration: {func_decl}")
    func_name = match.group(1).strip()
    params_str = match.group(2).strip()
    returns_str = match.group(3).strip()
    params = []
    for token in split_top_level(params_str):
        parsed = parse_param(token)
        if parsed is not None:
            params.append(parsed)
    returns = parse_returns(returns_str)
    overload_signature = f"({params_str})"
    return func_name, overload_signature, params, returns


def pta_api_from_func_name(func_name: str) -> str:
    root = normalize_root_name(func_name)
    if root.startswith("npu_"):
        return f"torch_npu.{root}"
    return f"torch.{root}"


def is_opapi_relevant(entry: dict) -> bool:
    return "op_api" in entry or "gen_opapi" in entry


def load_functions_entries(path: Path) -> List[dict]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows: Dict[str, dict] = {}
    if not isinstance(obj, dict):
        return []
    for section, items in obj.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or "func" not in item:
                continue
            func_decl = str(item["func"]).strip()
            try:
                func_name, overload_signature, params, returns = parse_func_decl(func_decl)
            except ValueError:
                continue
            key = f"{func_name}::{overload_signature}"
            merged = rows.setdefault(
                key,
                {
                    "section": section,
                    "func_decl": func_decl,
                    "func_name": func_name,
                    "overload_signature": overload_signature,
                    "params": params,
                    "returns": returns,
                    "raw_entries": [],
                    "has_opapi": False,
                    "gen_opapi": None,
                },
            )
            merged["raw_entries"].append(item)
            if is_opapi_relevant(item):
                merged["has_opapi"] = True
            if isinstance(item.get("gen_opapi"), dict) and merged["gen_opapi"] is None:
                merged["gen_opapi"] = item.get("gen_opapi")
    return list(rows.values())


def build_func_lookup(entries: Iterable[dict]) -> Dict[str, List[dict]]:
    lookup: Dict[str, List[dict]] = {}
    for entry in entries:
        lookup.setdefault(entry["func_name"], []).append(entry)
    return lookup


def resolve_gen_opapi_exec(entry: dict, lookup: Dict[str, List[dict]], seen: Optional[set] = None) -> Optional[str]:
    if seen is None:
        seen = set()
    key = entry["func_name"]
    if key in seen:
        return None
    seen.add(key)
    gen = entry.get("gen_opapi")
    if not isinstance(gen, dict):
        return None
    exec_name = gen.get("exec")
    if isinstance(exec_name, str):
        exec_head = exec_name.split(",", 1)[0].strip()
        if GEN_EXEC_RE.match(exec_head):
            return exec_head
    inherit = gen.get("structured_inherit")
    if not isinstance(inherit, str) or not inherit:
        return None
    for candidate in lookup.get(inherit, []):
        resolved = resolve_gen_opapi_exec(candidate, lookup, seen)
        if resolved:
            return resolved
    return None


def load_derivatives(path: Path) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]]]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_decl: Dict[str, List[dict]] = {}
    by_root: Dict[str, List[dict]] = {}
    if not isinstance(obj, dict):
        return by_decl, by_root
    for item in obj.get("backward", []):
        if not isinstance(item, dict):
            continue
        name_decl = str(item.get("name", "")).strip()
        if not name_decl:
            continue
        name, _, _, _ = parse_func_decl(name_decl)
        formula_functions: List[str] = []
        differentiable_inputs: List[str] = []
        for key, value in item.items():
            if key in {"name", "version", "result", "output_differentiability"}:
                continue
            if not isinstance(value, str):
                continue
            if "non_differentiable" not in value:
                differentiable_inputs.extend([part.strip() for part in key.split(",") if part.strip()])
            func_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", value)
            if func_match:
                formula_functions.append(func_match.group(1))
        row = {
            "name_decl": name_decl,
            "name": name,
            "root_name": normalize_root_name(name),
            "differentiable_inputs": list(dict.fromkeys(differentiable_inputs)),
            "formula_functions": list(dict.fromkeys(formula_functions)),
        }
        by_decl.setdefault(name_decl, []).append(row)
        by_root.setdefault(row["root_name"], []).append(row)
    return by_decl, by_root


def opapi_file_aliases(path: Path) -> List[str]:
    stem = path.stem
    variants = {
        stem,
        re.sub(r"(KernelNpuOpApi|KernelOpApi|KernelNpu|KernelOp)$", "", stem),
        re.sub(r"(Backwrd|Backward)$", "", re.sub(r"(KernelNpuOpApi|KernelOpApi|KernelNpu|KernelOp)$", "", stem)),
    }
    return [normalize_token(item) for item in variants if item]


def load_opapi_files(roots: Iterable[Path]) -> Tuple[Dict[str, List[Path]], Dict[Path, str]]:
    by_alias: Dict[str, List[Path]] = {}
    text_map: Dict[Path, str] = {}
    seen_paths = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.cpp")):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            text = path.read_text(encoding="utf-8", errors="ignore")
            text_map[path] = text
            for alias in opapi_file_aliases(path):
                by_alias.setdefault(alias, []).append(path)
    return by_alias, text_map


def find_forward_file(entry: dict, by_alias: Dict[str, List[Path]], text_map: Dict[Path, str]) -> Optional[Path]:
    def is_backward_path(path: Path) -> bool:
        name = path.name.lower()
        return "backward" in name or "grad" in name

    candidates = []
    func_name = entry["func_name"]
    root = normalize_root_name(func_name)
    for token in {
        normalize_token(root),
        normalize_token(func_name),
        normalize_token(func_name.replace(".", "_")),
    }:
        candidates.extend(by_alias.get(token, []))
    candidates = sorted(dict.fromkeys(candidates), key=lambda path: (is_backward_path(path), str(path)))
    if candidates:
        return candidates[0]
    search_names = [
        func_name.replace(".out", "_out"),
        func_name.replace(".", "_"),
        root,
    ]
    for search_name in search_names:
        if not search_name:
            continue
        pattern = re.compile(
            rf"(?m)^\s*(?:[A-Za-z_][A-Za-z0-9_:<>\s*&]+)\b{re.escape(search_name)}\s*\("
        )
        matched = []
        for path, text in text_map.items():
            if pattern.search(text):
                matched.append(path)
        if matched:
            matched.sort(key=lambda path: (is_backward_path(path), str(path)))
            return matched[0]
    return None


def find_backward_file(formula_functions: List[str], text_map: Dict[Path, str]) -> Optional[Path]:
    def backward_score(path: Path) -> Tuple[int, str]:
        name = path.name.lower()
        preferred = 0 if ("backward" in name or "grad" in name) else 1
        return (preferred, str(path))

    for func_name in formula_functions:
        pattern = re.compile(rf"\b{re.escape(func_name)}\s*\(")
        matched = []
        for path, text in text_map.items():
            if pattern.search(text):
                matched.append(path)
        if matched:
            matched.sort(key=backward_score)
            return matched[0]
    return None


def filter_aclnn_hits(hits: Iterable[str], func_name: str) -> List[str]:
    unique = []
    seen = set()
    root_name = func_name.split(".", 1)[0]
    is_inplace = root_name.endswith("_")
    for hit in hits:
        if is_inplace and "Inplace" not in hit:
            continue
        if not is_inplace and "Inplace" in hit:
            continue
        if hit not in seen:
            seen.add(hit)
            unique.append(hit)
    if unique:
        return unique
    # fall back to unfiltered hits when only inplace variants exist in the file
    fallback = []
    seen.clear()
    for hit in hits:
        if hit not in seen:
            seen.add(hit)
            fallback.append(hit)
    return fallback


def canonical_aclnn_family(name: str) -> str:
    value = name
    if value.startswith("aclnn"):
        value = value[len("aclnn") :]
    value = re.sub(r"^Inplace", "", value)
    value = re.sub(r"V\d+$", "", value)
    for suffix in (
        "TensorScalar",
        "ScalarTensor",
        "TensorTensor",
        "Tensor",
        "Scalar",
        "Backward",
        "Grad",
    ):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    if value.endswith("s") and len(value) > 4:
        value = value[:-1]
    return value


def find_function_files(func_name: str, text_map: Dict[Path, str]) -> List[Path]:
    pattern = re.compile(rf"(?m)^\s*(?:[A-Za-z_][A-Za-z0-9_:<>\s*&]+)\b{re.escape(func_name)}\s*\(")
    matched = []
    for path, text in text_map.items():
        if pattern.search(text):
            matched.append(path)
    matched.sort(key=str)
    return matched


def resolve_delegate_aclnn(
    path: Path,
    text_map: Dict[Path, str],
    visited: Optional[set] = None,
    depth: int = 2,
) -> Tuple[List[str], List[Path]]:
    if visited is None:
        visited = set()
    if path in visited:
        return [], []
    visited.add(path)
    text = text_map.get(path, "")
    direct_hits = filter_aclnn_hits(EXEC_CMD_RE.findall(text), path.stem)
    if direct_hits:
        return direct_hits, []
    if depth <= 0:
        return [], []

    helper_hits: List[str] = []
    helper_paths: List[Path] = []
    for func_name in dict.fromkeys(DELEGATE_CALL_RE.findall(text)):
        for helper_path in find_function_files(func_name, text_map):
            if helper_path == path:
                continue
            hits, nested_paths = resolve_delegate_aclnn(helper_path, text_map, visited, depth - 1)
            if hits:
                helper_paths.append(helper_path)
                helper_paths.extend(nested_paths)
                helper_hits.extend(hits)
    dedup_hits = []
    seen_hits = set()
    for hit in helper_hits:
        if hit not in seen_hits:
            seen_hits.add(hit)
            dedup_hits.append(hit)
    dedup_paths = []
    seen_paths = set()
    for helper_path in helper_paths:
        if helper_path not in seen_paths:
            seen_paths.add(helper_path)
            dedup_paths.append(helper_path)
    return dedup_hits, dedup_paths


def infer_preprocess_needed(text: str, func_name: str, gen_exec: Optional[str]) -> bool:
    if gen_exec:
        return False
    return any(pattern in text for pattern in HELPER_PREPROCESS_PATTERNS)


def infer_custom_output_needed(text: str, returns: List[dict], gen_exec: Optional[str]) -> bool:
    if gen_exec:
        return False
    if len(returns) > 1:
        return True
    if "at::empty(" in text or "std::make_tuple(" in text:
        return True
    if any(pattern in text for pattern in CUSTOM_OUTPUT_PATTERNS):
        # simple one-output apply_tensor_without_format is common; keep it non-custom
        if len(returns) == 1 and "apply_tensor_without_format(" in text and "at::empty(" not in text:
            return False
        return True
    return False


def infer_composite(text: str, forward_aclnn: List[str], gen_exec: Optional[str]) -> bool:
    if gen_exec:
        return False
    if "op_api::" not in text:
        return False
    families = {canonical_aclnn_family(name) for name in forward_aclnn}
    return len(families) > 1


def build_refs(
    func_decl: str,
    derivatives: List[dict],
    forward_file: Optional[Path],
    forward_helper_files: List[Path],
    backward_file: Optional[Path],
    forward_aclnn: List[str],
    backward_aclnn: List[str],
    composite: bool,
) -> List[dict]:
    refs = [
        {
            "role": "forward_signature",
            "path": short_rel(DEFAULT_FUNCTIONS_YAML),
            "pattern": f"func: {func_decl}",
        }
    ]
    if derivatives:
        refs.append(
            {
                "role": "backward_registration",
                "path": short_rel(DEFAULT_DERIVATIVES_YAML),
                "pattern": f"name: {derivatives[0]['name_decl']}",
            }
        )
    if forward_file is not None:
        refs.append(
            {
                "role": "forward_logic",
                "path": short_rel(forward_file),
                "pattern": forward_aclnn[0] if forward_aclnn else "EXEC_NPU_CMD",
            }
        )
    for helper_file in forward_helper_files:
        refs.append(
            {
                "role": "forward_logic_helper",
                "path": short_rel(helper_file),
                "pattern": forward_aclnn[0] if forward_aclnn else "EXEC_NPU_CMD",
            }
        )
    if backward_file is not None:
        refs.append(
            {
                "role": "backward_logic",
                "path": short_rel(backward_file),
                "pattern": backward_aclnn[0] if backward_aclnn else "EXEC_NPU_CMD",
            }
        )
    if composite and forward_file is not None:
        refs.append(
            {
                "role": "callchain_logic",
                "path": short_rel(forward_file),
                "pattern": "EXEC_NPU_CMD",
            }
        )
    return refs


def build_pta_rows(op_plugin_root: Path) -> List[dict]:
    functions_yaml = op_plugin_root / "op_plugin" / "config" / "op_plugin_functions.yaml"
    derivatives_yaml = op_plugin_root / "op_plugin" / "config" / "derivatives.yaml"
    opapi_root = op_plugin_root / "op_plugin" / "ops" / "opapi"
    helper_root = op_plugin_root / "op_plugin" / "utils" / "custom_functions" / "opapi"

    entries = load_functions_entries(functions_yaml)
    func_lookup = build_func_lookup(entries)
    deriv_by_decl, deriv_by_root = load_derivatives(derivatives_yaml)
    file_aliases, text_map = load_opapi_files([opapi_root, helper_root])

    rows: List[dict] = []
    for entry in entries:
        forward_file = find_forward_file(entry, file_aliases, text_map)
        derivatives = deriv_by_decl.get(entry["func_decl"], []) or deriv_by_root.get(normalize_root_name(entry["func_name"]), [])
        has_relevance = entry["has_opapi"] or entry.get("gen_opapi") is not None or forward_file is not None or bool(derivatives)
        if not has_relevance:
            continue

        gen_exec = resolve_gen_opapi_exec(entry, func_lookup)
        forward_text = text_map.get(forward_file, "") if forward_file is not None else ""
        forward_hits = [gen_exec] if gen_exec else EXEC_CMD_RE.findall(forward_text)
        forward_helper_files: List[Path] = []
        if not gen_exec and forward_file is not None and not forward_hits:
            delegated_hits, delegated_files = resolve_delegate_aclnn(forward_file, text_map, depth=2)
            forward_hits.extend(delegated_hits)
            forward_helper_files = delegated_files
        helper_text = "\n".join(text_map.get(path, "") for path in forward_helper_files)
        effective_forward_text = "\n".join(part for part in [forward_text, helper_text] if part)
        forward_aclnn = filter_aclnn_hits(forward_hits, entry["func_name"])

        formula_functions = []
        for item in derivatives:
            formula_functions.extend(item.get("formula_functions", []))
        formula_functions = list(dict.fromkeys(formula_functions))
        backward_file = find_backward_file(formula_functions, text_map) if formula_functions else None
        backward_text = text_map.get(backward_file, "") if backward_file is not None else ""
        backward_aclnn = filter_aclnn_hits(EXEC_CMD_RE.findall(backward_text), entry["func_name"])

        composite = infer_composite(effective_forward_text, forward_aclnn, gen_exec)
        preprocess_needed = infer_preprocess_needed(effective_forward_text, entry["func_name"], gen_exec)
        custom_output_needed = infer_custom_output_needed(effective_forward_text, entry["returns"], gen_exec)
        pta_api = pta_api_from_func_name(entry["func_name"])
        pta_key = f"{pta_api}::{entry['overload_signature']}"

        refs = build_refs(
            func_decl=entry["func_decl"],
            derivatives=derivatives,
            forward_file=forward_file,
            forward_helper_files=forward_helper_files,
            backward_file=backward_file,
            forward_aclnn=forward_aclnn,
            backward_aclnn=backward_aclnn,
            composite=composite,
        )

        rows.append(
            {
                "pta_key": pta_key,
                "pta_api": pta_api,
                "overload_signature": entry["overload_signature"],
                "params": entry["params"],
                "returns": entry["returns"],
                "forward_aclnn": forward_aclnn,
                "backward_exists": bool(derivatives),
                "backward_aclnn": backward_aclnn,
                "composite": composite,
                "preprocess_needed": preprocess_needed,
                "custom_output_needed": custom_output_needed,
                "refs": refs,
            }
        )
    rows.sort(key=lambda item: item["pta_key"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build operator-facts PTA facts from op-plugin")
    parser.add_argument("--op-plugin-root", type=Path, default=DEFAULT_OP_PLUGIN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = build_pta_rows(args.op_plugin_root.resolve())
    jsonl_path = args.out_dir / "pta_facts.jsonl"
    csv_path = args.out_dir / "pta_facts.csv"
    write_jsonl(jsonl_path, rows)
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "pta_key": row["pta_key"],
                "pta_api": row["pta_api"],
                "overload_signature": row["overload_signature"],
                "params": json.dumps(row["params"], ensure_ascii=False),
                "returns": json.dumps(row["returns"], ensure_ascii=False),
                "forward_aclnn": json.dumps(row["forward_aclnn"], ensure_ascii=False),
                "backward_exists": str(row["backward_exists"]).lower(),
                "backward_aclnn": json.dumps(row["backward_aclnn"], ensure_ascii=False),
                "composite": str(row["composite"]).lower(),
                "preprocess_needed": str(row["preprocess_needed"]).lower(),
                "custom_output_needed": str(row["custom_output_needed"]).lower(),
                "refs": json.dumps(row["refs"], ensure_ascii=False),
            }
        )
    write_csv(
        csv_path,
        csv_rows,
        [
            "pta_key",
            "pta_api",
            "overload_signature",
            "params",
            "returns",
            "forward_aclnn",
            "backward_exists",
            "backward_aclnn",
            "composite",
            "preprocess_needed",
            "custom_output_needed",
            "refs",
        ],
    )
    print(f"pta_facts_rows={len(rows)}")


if __name__ == "__main__":
    main()
