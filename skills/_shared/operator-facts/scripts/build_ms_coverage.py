from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Dict, Iterable, List, Set, Tuple

from common import (
    DEFAULT_MS_ROOT,
    DEFAULT_OUT_DIR,
    OpDefEntry,
    build_alias_keys,
    build_op_catalog,
    json_array_string,
    load_yaml_obj,
    write_csv,
    write_jsonl,
)


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
PYBOOST_CUSTOMIZE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)AscendCustomize\b")


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


def scan_path_matches(roots: Iterable[Path], ms_root: Path, suffixes: Tuple[str, ...]) -> Dict[str, Set[str]]:
    hits: Dict[str, Set[str]] = {}
    for path in collect_text_files(roots, suffixes):
        rel = short_rel(path, ms_root)
        add_file(hits, build_alias_keys(path.stem), rel)
    return hits


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build phase-1 ms_coverage index for operator-facts.")
    parser.add_argument("--ms-root", type=Path, default=DEFAULT_MS_ROOT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_DIR / "ms_coverage.jsonl")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_DIR / "ms_coverage.csv")
    args = parser.parse_args()

    op_by_branch, _ = build_op_catalog(args.ms_root)
    aclnn_config = load_aclnn_config(args.ms_root)
    customize_kernel_map, customize_kernel_files = scan_kernel_customize(args.ms_root)
    auto_kernel_map, auto_kernel_files = scan_kernel_auto_generate(args.ms_root)
    customize_pyboost_map, customize_pyboost_files = scan_pyboost_customize(args.ms_root)
    auto_pyboost_files = scan_pyboost_auto_generate(args.ms_root)
    infer_primitives, infer_files = scan_infer(args.ms_root)
    bprop_files = scan_bprop(args.ms_root)
    ut_index = scan_path_matches(
        [args.ms_root.parent / "tests" / "ut" / "cpp" / "ops"],
        args.ms_root,
        (".cc", ".h", ".hpp"),
    )
    st_index = scan_path_matches(
        [args.ms_root.parent / "tests" / "st" / "ops"],
        args.ms_root,
        (".py", ".cc", ".h", ".hpp", ".rst"),
    )
    cn_docs_index = scan_path_matches(
        [args.ms_root.parent / "docs" / "api" / "api_python"],
        args.ms_root,
        (".rst", ".yaml", ".md"),
    )
    en_docs_index = scan_path_matches(
        [args.ms_root.parent / "docs" / "api" / "api_python_en"],
        args.ms_root,
        (".rst", ".yaml", ".md", ".txt"),
    )
    function_doc_index = scan_path_matches(
        [args.ms_root / "ops" / "api_def" / "function_doc"],
        args.ms_root,
        (".yaml",),
    )
    op_doc_index = scan_path_matches(
        [args.ms_root / "ops" / "op_def" / "yaml" / "doc"],
        args.ms_root,
        (".yaml",),
    )

    rows: List[dict] = []
    for entry in sorted(op_by_branch.values(), key=lambda item: (item.op, item.primitive)):
        candidate_keys = build_candidate_keys(entry)
        dispatch_kind = resolve_dispatch_kind(entry)
        aclnn_values: Set[str] = set()
        aclnn_source: List[str] = []

        if dispatch_kind == "customize":
            for key in candidate_keys:
                aclnn_values.update(customize_kernel_map.get(key, set()))
                aclnn_values.update(customize_pyboost_map.get(key, set()))
            if aclnn_values:
                aclnn_source.append("customize_source")
            pyboost_evidence = match_named_files(candidate_keys, customize_pyboost_files)
            kbk_evidence = match_named_files(candidate_keys, customize_kernel_files)
        elif dispatch_kind == "auto_generate":
            for key in candidate_keys:
                mapped = aclnn_config.get(key)
                if mapped:
                    aclnn_values.add(mapped)
                aclnn_values.update(auto_kernel_map.get(key, set()))
            if any(aclnn_config.get(key) for key in candidate_keys):
                aclnn_source.append("aclnn_config")
            if any(auto_kernel_map.get(key) for key in candidate_keys):
                aclnn_source.append("auto_generate_source")
            if not aclnn_values and entry.dispatch_enable:
                aclnn_values.add(f"aclnn{entry.class_name}")
                aclnn_source.append("default_guess")
            pyboost_evidence = match_named_files(candidate_keys, auto_pyboost_files)
            kbk_evidence = match_named_files(candidate_keys, auto_kernel_files)
        else:
            pyboost_evidence = []
            kbk_evidence = []

        infer_evidence = match_named_files(candidate_keys, infer_files)
        infer_present = entry.primitive in infer_primitives or bool(infer_evidence)
        bprop_evidence = match_named_files(build_alias_keys(entry.primitive), bprop_files)
        ut_evidence = match_named_files(candidate_keys, ut_index)
        st_evidence = match_named_files(candidate_keys, st_index)
        cn_docs_evidence = match_named_files(candidate_keys, cn_docs_index)
        en_docs_evidence = sorted(
            set(match_named_files(candidate_keys, en_docs_index))
            | set(match_named_files(candidate_keys, function_doc_index))
            | set(match_named_files(candidate_keys, op_doc_index))
        )

        rows.append(
            {
                "coverage_key": f"{entry.op}::{entry.primitive}",
                "op": entry.op,
                "primitive": entry.primitive,
                "op_branch": entry.op_branch,
                "op_yaml_path": entry.op_yaml_path,
                "class_name": entry.class_name,
                "dispatch_enable": str(entry.dispatch_enable),
                "dispatch_kind": dispatch_kind,
                "dispatch_ascend": entry.dispatch_ascend,
                "aclnn": json_array_string(sorted(aclnn_values)),
                "aclnn_source": ",".join(sorted(set(aclnn_source))),
                "infer": "true" if infer_present else "false",
                "pyboost": "true" if bool(pyboost_evidence) else "false",
                "kbk": "true" if bool(kbk_evidence) else "false",
                "bprop": "true" if bool(bprop_evidence) else "false",
                "ut": "true" if bool(ut_evidence) else "false",
                "st": "true" if bool(st_evidence) else "false",
                "docs_cn": "true" if bool(cn_docs_evidence) else "false",
                "docs_en": "true" if bool(en_docs_evidence) else "false",
                "infer_evidence": json_array_string(infer_evidence),
                "pyboost_evidence": json_array_string(pyboost_evidence),
                "kbk_evidence": json_array_string(kbk_evidence),
                "bprop_evidence": json_array_string(bprop_evidence),
                "ut_evidence": json_array_string(ut_evidence),
                "st_evidence": json_array_string(st_evidence),
                "docs_cn_evidence": json_array_string(cn_docs_evidence),
                "docs_en_evidence": json_array_string(en_docs_evidence),
            }
        )

    write_jsonl(args.out_jsonl, rows)
    fieldnames = [
        "coverage_key",
        "op",
        "primitive",
        "op_branch",
        "op_yaml_path",
        "class_name",
        "dispatch_enable",
        "dispatch_kind",
        "dispatch_ascend",
        "aclnn",
        "aclnn_source",
        "infer",
        "pyboost",
        "kbk",
        "bprop",
        "ut",
        "st",
        "docs_cn",
        "docs_en",
        "infer_evidence",
        "pyboost_evidence",
        "kbk_evidence",
        "bprop_evidence",
        "ut_evidence",
        "st_evidence",
        "docs_cn_evidence",
        "docs_en_evidence",
    ]
    write_csv(args.out_csv, rows, fieldnames)
    print(f"ms_coverage_rows={len(rows)}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"out_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
