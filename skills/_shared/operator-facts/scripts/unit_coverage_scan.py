from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from common import OpDefEntry, build_alias_keys, load_yaml_obj, normalize_token


ACLNN_API_RE = re.compile(r"\baclnn[A-Z][A-Za-z0-9_]*\b")
LAUNCH_ACLNN_RE = re.compile(r"\bLAUNCH_ACLNN(?:_SYNC)?\s*\(\s*(aclnn[A-Z][A-Za-z0-9_]*)\b")
COMMON_MACRO_RE = re.compile(
    r"\bMS_ACLNN_COMMON_KERNEL_FACTORY_REG\s*\(\s*([A-Za-z0-9_]+)\s*,\s*(aclnn[A-Z][A-Za-z0-9_]*)\s*,"
)
KERNEL_REG_RE = re.compile(
    r"\bMS_ACLNN_KERNEL_FACTORY_REG\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"
)
CTOR_ACLNN_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*:\s*AclnnKernelMod\([^)]*\"(aclnn[A-Z][A-Za-z0-9_]*)\"",
    re.S,
)
WS_ACLNN_RE = re.compile(r"\bDEFINE_GET_WORKSPACE_FOR_OPS\s*\(\s*(aclnn[A-Z][A-Za-z0-9_]*)\b")
CLASS_DEF_RE = re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b")
INFER_REG_RE = re.compile(r"REGISTER_PRIMITIVE_OP_INFER_IMPL\(\s*([A-Za-z0-9_]+)\s*,")
BPROP_RE = re.compile(r'REG_BPROP_BUILDER\("([A-Za-z0-9_]+)"\)')
EMIT_SYMBOL_RE = re.compile(r'\bEmit\s*\(\s*"([A-Za-z0-9_]+)"')
EMIT_IDENT_RE = re.compile(r'\bEmit\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)')
BUILDER_METHOD_RE = re.compile(r'\bib->([A-Za-z_][A-Za-z0-9_]*)\s*\(')
MUL_OP_RE = re.compile(r'(?<![/*<>=!+\-])\*(?![/*=])')
PYBOOST_CUSTOMIZE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)AscendCustomize\b")

LEFT_OPERAND_CHARS = set(")]}_\"'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
RIGHT_OPERAND_CHARS = set("([{\"'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")


def short_rel(path: Path, ms_root: Path) -> str:
    try:
        return str(path.relative_to(ms_root.parent))
    except ValueError:
        return str(path)


def collect_text_files(roots: Iterable[Path], suffixes: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    seen: Set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for suffix in suffixes:
            for path in sorted(root.rglob(f"*{suffix}")):
                if path in seen:
                    continue
                seen.add(path)
                files.append(path)
    return files


def add_values(index: Dict[str, Set[str]], keys: Iterable[str], values: Iterable[str]) -> None:
    values = {value for value in values if value}
    if not values:
        return
    for key in keys:
        if not key:
            continue
        index.setdefault(key, set()).update(values)


def add_file(index: Dict[str, Set[str]], keys: Iterable[str], rel: str) -> None:
    if not rel:
        return
    for key in keys:
        if not key:
            continue
        index.setdefault(key, set()).add(rel)


def file_aliases(path: Path, text: str, include_classes: bool = False) -> Set[str]:
    aliases = set(build_alias_keys(path.stem))
    for op_name, _ in COMMON_MACRO_RE.findall(text):
        aliases.update(build_alias_keys(op_name))
    for op_name, class_name in KERNEL_REG_RE.findall(text):
        aliases.update(build_alias_keys(op_name))
        aliases.update(build_alias_keys(class_name))
    for symbol in PYBOOST_CUSTOMIZE_RE.findall(text):
        aliases.update(build_alias_keys(symbol))
    if include_classes:
        for class_name in CLASS_DEF_RE.findall(text):
            aliases.update(build_alias_keys(class_name))
    return aliases


def load_aclnn_config(ms_root: Path) -> Dict[str, str]:
    path = ms_root / "python" / "mindspore" / "ops_generate" / "pyboost" / "aclnn_config.yaml"
    obj = load_yaml_obj(path)
    mapping: Dict[str, str] = {}
    for key, value in obj.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        for alias in build_alias_keys(key):
            mapping[alias] = value
    return mapping


def collect_class_aclnn(files: Iterable[Path], ms_root: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    class_to_aclnn: Dict[str, Set[str]] = {}
    class_to_files: Dict[str, Set[str]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = short_rel(path, ms_root)
        for class_name, aclnn in CTOR_ACLNN_RE.findall(text):
            class_to_aclnn.setdefault(class_name, set()).add(aclnn)
            class_to_files.setdefault(class_name, set()).add(rel)

        current_class = None
        brace_depth = 0
        brace_started = False
        for line in text.splitlines():
            if current_class is None:
                match = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
                if match:
                    current_class = match.group(1)
                    brace_depth = line.count("{") - line.count("}")
                    brace_started = brace_depth > 0
                continue
            hits = WS_ACLNN_RE.findall(line)
            if hits:
                class_to_aclnn.setdefault(current_class, set()).update(hits)
                class_to_files.setdefault(current_class, set()).add(rel)
            brace_depth += line.count("{") - line.count("}")
            if not brace_started and "{" in line:
                brace_started = True
            if brace_started and brace_depth <= 0:
                current_class = None
                brace_depth = 0
                brace_started = False
    return class_to_aclnn, class_to_files


def scan_kernel_customize(ms_root: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    root = ms_root / "ops" / "kernel" / "ascend" / "aclnn" / "kernel_mod_impl" / "customize"
    files = collect_text_files([root], (".h", ".hpp", ".cc", ".cpp"))
    class_to_aclnn, class_to_files = collect_class_aclnn(files, ms_root)
    name_to_aclnn: Dict[str, Set[str]] = {}
    name_to_files: Dict[str, Set[str]] = {}

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = short_rel(path, ms_root)
        aliases = file_aliases(path, text, include_classes=False)
        file_aclnn_hits = set(ACLNN_API_RE.findall(text))
        add_file(name_to_files, aliases, rel)

        for op_name, aclnn in COMMON_MACRO_RE.findall(text):
            op_aliases = build_alias_keys(op_name)
            add_values(name_to_aclnn, op_aliases, [aclnn])
            add_file(name_to_files, op_aliases, rel)

        for op_name, class_name in KERNEL_REG_RE.findall(text):
            reg_aliases = set(build_alias_keys(op_name))
            reg_aliases.update(build_alias_keys(class_name))
            add_file(name_to_files, reg_aliases, rel)
            mapped_aclnn = set(class_to_aclnn.get(class_name, set()))
            if not mapped_aclnn and file_aclnn_hits:
                mapped_aclnn = set(file_aclnn_hits)
            add_values(name_to_aclnn, reg_aliases, mapped_aclnn)
            for file_rel in class_to_files.get(class_name, set()):
                add_file(name_to_files, reg_aliases, file_rel)

        stem_aliases = build_alias_keys(path.stem)
        if file_aclnn_hits and len(file_aclnn_hits) <= 4:
            add_values(name_to_aclnn, stem_aliases, file_aclnn_hits)

    return name_to_aclnn, name_to_files


def scan_kernel_auto_generate(ms_root: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    data_root = ms_root / "ops" / "kernel" / "ascend" / "aclnn" / "kernel_mod_impl" / "aclnn_auto_gen"
    reg_file = ms_root / "ops" / "kernel" / "ascend" / "aclnn" / "kernel_mod_impl" / "auto_generate" / "aclnn_kernel_register_auto.cc"
    files = collect_text_files([data_root], (".h", ".hpp", ".cc", ".cpp"))
    name_to_aclnn: Dict[str, Set[str]] = {}
    name_to_files: Dict[str, Set[str]] = {}

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = short_rel(path, ms_root)
        aliases = file_aliases(path, text, include_classes=False)
        aclnn_hits = set(ACLNN_API_RE.findall(text))
        add_file(name_to_files, aliases, rel)
        if aclnn_hits and len(aclnn_hits) <= 4:
            add_values(name_to_aclnn, aliases, aclnn_hits)

        for op_name, class_name in KERNEL_REG_RE.findall(text):
            reg_aliases = set(build_alias_keys(op_name))
            reg_aliases.update(build_alias_keys(class_name))
            add_file(name_to_files, reg_aliases, rel)
            if aclnn_hits:
                add_values(name_to_aclnn, reg_aliases, aclnn_hits)

    if reg_file.exists():
        rel = short_rel(reg_file, ms_root)
        text = reg_file.read_text(encoding="utf-8", errors="ignore")
        for op_name, aclnn in COMMON_MACRO_RE.findall(text):
            op_aliases = build_alias_keys(op_name)
            add_values(name_to_aclnn, op_aliases, [aclnn])
            add_file(name_to_files, op_aliases, rel)

    return name_to_aclnn, name_to_files


def scan_pyboost_customize(ms_root: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    roots = [
        ms_root / "ops" / "kernel" / "ascend" / "aclnn" / "pyboost_impl" / "customize",
        ms_root / "ops" / "kernel" / "ascend" / "aclnn" / "pyboost_impl" / "internal" / "customize",
    ]
    name_to_aclnn: Dict[str, Set[str]] = {}
    name_to_files: Dict[str, Set[str]] = {}
    for path in collect_text_files(roots, (".h", ".hpp", ".cc", ".cpp")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = short_rel(path, ms_root)
        aliases = file_aliases(path, text, include_classes=False)
        add_file(name_to_files, aliases, rel)
        aclnn_hits = set(LAUNCH_ACLNN_RE.findall(text))
        if aclnn_hits:
            add_values(name_to_aclnn, aliases, aclnn_hits)
    return name_to_aclnn, name_to_files


def scan_pyboost_auto_generate(ms_root: Path) -> Dict[str, Set[str]]:
    roots = [
        ms_root / "ops" / "kernel" / "ascend" / "aclnn" / "pyboost_impl" / "auto_generate",
        ms_root / "ops" / "kernel" / "ascend" / "aclnn" / "pyboost_impl" / "internal" / "auto_generate",
    ]
    name_to_files: Dict[str, Set[str]] = {}
    for path in collect_text_files(roots, (".h", ".hpp")):
        rel = short_rel(path, ms_root)
        add_file(name_to_files, build_alias_keys(path.stem), rel)
    return name_to_files


def scan_infer(ms_root: Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    root = ms_root / "ops" / "infer"
    primitive_hits: Set[str] = set()
    file_hits: Dict[str, Set[str]] = {}
    for path in collect_text_files([root], (".cc", ".h", ".hpp")):
        rel = short_rel(path, ms_root)
        text = path.read_text(encoding="utf-8", errors="ignore")
        add_file(file_hits, build_alias_keys(path.stem), rel)
        for primitive in INFER_REG_RE.findall(text):
            primitive_hits.add(primitive)
            add_file(file_hits, build_alias_keys(primitive), rel)
    return primitive_hits, file_hits


def scan_bprop(ms_root: Path) -> Dict[str, Set[str]]:
    root = ms_root / "ccsrc" / "frontend" / "expander" / "grad"
    primitive_to_files: Dict[str, Set[str]] = {}
    for path in collect_text_files([root], (".cc", ".h", ".hpp")):
        rel = short_rel(path, ms_root)
        text = path.read_text(encoding="utf-8", errors="ignore")
        for primitive in BPROP_RE.findall(text):
            add_file(primitive_to_files, build_alias_keys(primitive), rel)
    return primitive_to_files


def trim_bprop_symbol(name: str) -> Optional[str]:
    if not name:
        return None
    trimmed = name
    if trimmed.startswith("k") and len(trimmed) > 1 and trimmed[1].isupper():
        trimmed = trimmed[1:]
    for suffix in ("OpName", "Primitive", "Prim"):
        if trimmed.endswith(suffix):
            trimmed = trimmed[: -len(suffix)]
    return trimmed or None


def find_builder_body(text: str, start: int) -> Optional[Tuple[str, int]]:
    set_body_pos = text.find("SetBody", start)
    search_start = set_body_pos if set_body_pos >= 0 else start
    brace_pos = text.find("{", search_start)
    if brace_pos < 0:
        return None
    depth = 0
    for idx in range(brace_pos, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_pos + 1 : idx], idx + 1
    return None


def collect_bprop_symbols(body: str) -> Set[str]:
    symbols = set(EMIT_SYMBOL_RE.findall(body))
    symbols.update(EMIT_IDENT_RE.findall(body))
    symbols.update(BUILDER_METHOD_RE.findall(body))
    # `a * b` does not surface as `ib->Mul(...)`, but for bprop purposes we still
    # want to record that a `Mul` unit participates in the formula.
    if MUL_OP_RE.search(body):
        symbols.add("Mul")
    if contains_binary_operator(body, "+"):
        symbols.add("Add")
    if contains_binary_operator(body, "-"):
        symbols.add("Sub")
    return symbols


def contains_binary_operator(text: str, operator: str) -> bool:
    for index, char in enumerate(text):
        if char != operator:
            continue
        prev_char = previous_non_space_char(text, index)
        next_char = next_non_space_char(text, index)
        if prev_char is None or next_char is None:
            continue
        if operator == "+" and next_char == "+":
            continue
        if operator == "-" and next_char == "-":
            continue
        if next_char == "=":
            continue
        if prev_char not in LEFT_OPERAND_CHARS:
            continue
        if next_char not in RIGHT_OPERAND_CHARS:
            continue
        return True
    return False


def previous_non_space_char(text: str, index: int) -> Optional[str]:
    pos = index - 1
    while pos >= 0:
        char = text[pos]
        if not char.isspace():
            return char
        pos -= 1
    return None


def next_non_space_char(text: str, index: int) -> Optional[str]:
    pos = index + 1
    while pos < len(text):
        char = text[pos]
        if not char.isspace():
            return char
        pos += 1
    return None


def resolve_bprop_unit_names(symbols: Iterable[str], op_by_symbol: Dict[str, List[OpDefEntry]]) -> List[str]:
    unit_names: Set[str] = set()
    for symbol in symbols:
        candidate_names = [symbol]
        trimmed = trim_bprop_symbol(symbol)
        if trimmed and trimmed != symbol:
            candidate_names.append(trimmed)
        seen_keys: Set[str] = set()
        for candidate in candidate_names:
            exact_matches: List[OpDefEntry] = []
            fallback_matches: List[OpDefEntry] = []
            candidate_token = normalize_token(candidate)
            for key in build_alias_keys(candidate):
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                fallback_matches.extend(op_by_symbol.get(key, []))
            for entry in fallback_matches:
                exact_fields = {
                    normalize_token(entry.op),
                    normalize_token(entry.primitive),
                    normalize_token(entry.class_name),
                    normalize_token(Path(entry.op_branch).stem),
                    normalize_token(Path(entry.op_branch).stem.removesuffix("_op")),
                }
                if candidate_token in exact_fields:
                    exact_matches.append(entry)
            selected = exact_matches or fallback_matches
            for entry in selected:
                unit_names.add(entry.primitive)
    return sorted(unit_names)


def scan_bprop_units(ms_root: Path, op_by_symbol: Dict[str, List[OpDefEntry]]) -> Dict[str, Set[str]]:
    root = ms_root / "ccsrc" / "frontend" / "expander" / "grad"
    primitive_to_units: Dict[str, Set[str]] = {}
    for path in collect_text_files([root], (".cc", ".h", ".hpp")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        pos = 0
        while True:
            match = BPROP_RE.search(text, pos)
            if match is None:
                break
            primitive = match.group(1)
            body_info = find_builder_body(text, match.end())
            if body_info is None:
                pos = match.end()
                continue
            body, end_pos = body_info
            symbols = collect_bprop_symbols(body)
            unit_names = resolve_bprop_unit_names(symbols, op_by_symbol)
            if unit_names:
                for key in build_alias_keys(primitive):
                    primitive_to_units.setdefault(key, set()).update(unit_names)
            pos = end_pos
    return primitive_to_units


def build_candidate_keys(entry: OpDefEntry) -> Set[str]:
    values = {
        entry.op,
        entry.primitive,
        entry.class_name,
        Path(entry.op_branch).stem,
        Path(entry.op_branch).stem.removesuffix("_op"),
    }
    if entry.dispatch_ascend:
        values.add(entry.dispatch_ascend)
        if entry.dispatch_ascend.endswith("Ascend"):
            values.add(entry.dispatch_ascend.removesuffix("Ascend"))
    keys: Set[str] = set()
    for value in values:
        keys.update(build_alias_keys(value))
    return keys


def match_named_files(candidate_keys: Iterable[str], index: Dict[str, Set[str]]) -> List[str]:
    files: Set[str] = set()
    for key in candidate_keys:
        files.update(index.get(key, set()))
    return sorted(files)


def resolve_dispatch_kind(entry: OpDefEntry) -> str:
    if not entry.dispatch_enable:
        return "unsupported"
    ascend = entry.dispatch_ascend
    if ascend and ascend not in {"default", "None"}:
        return "customize"
    return "auto_generate"
