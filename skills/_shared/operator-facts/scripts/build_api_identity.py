from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from common import (
    DEFAULT_MS_ROOT,
    DEFAULT_OUT_DIR,
    MINT_INIT,
    ApiBranchEntry,
    OpDefEntry,
    build_api_def_catalog,
    build_op_catalog,
    iter_return_calls,
    load_module_info,
    normalize_token,
    write_csv,
    write_jsonl,
    extract_call_target,
)


def resolve_symbol_to_entries(
    ms_root: Path,
    target_module: str,
    target_symbol: str,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_symbol: Dict[str, List[OpDefEntry]],
    visited: Optional[Set[Tuple[str, str]]] = None,
    depth: int = 0,
) -> List[dict]:
    if visited is None:
        visited = set()
    state = (target_module, target_symbol)
    if state in visited or depth > 4:
        return []
    visited = set(visited)
    visited.add(state)

    if target_symbol in api_def_catalog:
        rows = []
        for branch in api_def_catalog[target_symbol]:
            if branch.deprecated:
                continue
            rows.append(
                {
                    "op_branch": branch.op_branch,
                    "py_method": branch.py_method,
                    "interface": branch.interface,
                    "resolver_kind": "api_def",
                    "resolver_path": f"{target_module}.{target_symbol}",
                }
            )
        if rows:
            return rows

    candidates = [
        target_symbol,
        target_symbol.removesuffix("_"),
        target_symbol.removesuffix("_impl"),
    ]
    seen = set()
    direct_rows = []
    for candidate in candidates:
        key = normalize_token(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        for entry in op_by_symbol.get(key, []):
            direct_rows.append(
                {
                    "op_branch": entry.op_branch,
                    "py_method": "",
                    "interface": "",
                    "resolver_kind": "op_def",
                    "resolver_path": f"{target_module}.{target_symbol}",
                }
            )
    if direct_rows:
        dedup = {}
        for row in direct_rows:
            dedup[row["op_branch"]] = row
        return list(dedup.values())

    info = load_module_info(ms_root, target_module)
    if info is None:
        return []
    func = info.functions.get(target_symbol)
    if func is None:
        return []
    rows = []
    for call in iter_return_calls(func):
        target = extract_call_target(call)
        if target is None:
            continue
        base, attr = target
        if attr is None:
            if base in info.from_imports:
                binding = info.from_imports[base]
                rows.extend(
                    resolve_symbol_to_entries(
                        ms_root,
                        binding.module,
                        binding.symbol or base,
                        api_def_catalog,
                        op_by_symbol,
                        visited=visited,
                        depth=depth + 1,
                    )
                )
            else:
                rows.extend(
                    resolve_symbol_to_entries(
                        ms_root,
                        target_module,
                        base,
                        api_def_catalog,
                        op_by_symbol,
                        visited=visited,
                        depth=depth + 1,
                    )
                )
        else:
            if base in info.imports:
                module_name = info.imports[base]
                rows.extend(
                    resolve_symbol_to_entries(
                        ms_root,
                        module_name,
                        attr,
                        api_def_catalog,
                        op_by_symbol,
                        visited=visited,
                        depth=depth + 1,
                    )
                )
            elif base in info.from_imports:
                binding = info.from_imports[base]
                nested_module = binding.module
                rows.extend(
                    resolve_symbol_to_entries(
                        ms_root,
                        nested_module,
                        attr,
                        api_def_catalog,
                        op_by_symbol,
                        visited=visited,
                        depth=depth + 1,
                    )
                )
            else:
                rows.extend(
                    resolve_symbol_to_entries(
                        ms_root,
                        target_module,
                        attr,
                        api_def_catalog,
                        op_by_symbol,
                        visited=visited,
                        depth=depth + 1,
                    )
                )
    dedup = {}
    for row in rows:
        dedup[row["op_branch"]] = row
    return list(dedup.values())


def build_ops_and_tensor_rows(
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_branch: Dict[str, OpDefEntry],
) -> List[dict]:
    rows = []
    for api_name, branches in sorted(api_def_catalog.items()):
        for branch in branches:
            if branch.deprecated:
                continue
            op_entry = op_by_branch.get(branch.op_branch)
            if op_entry is None:
                continue
            interfaces = {item.strip() for item in branch.interface.split(",") if item.strip()}
            for surface in sorted(interfaces):
                if surface == "function":
                    public_api = f"mindspore.ops.{api_name}"
                    rows.append(
                        {
                            "identity_key": f"{public_api}::{branch.op_branch}",
                            "public_api": public_api,
                            "public_surface": "mindspore.ops",
                            "api_name": api_name,
                            "op_branch": branch.op_branch,
                            "op": op_entry.op,
                            "primitive": op_entry.primitive,
                            "py_method": branch.py_method,
                            "interface": branch.interface,
                            "target_module": "mindspore.ops.api_def",
                            "target_symbol": api_name,
                            "resolver_kind": "api_def_interface",
                            "resolver_path": branch.source_file,
                            "source_file": branch.source_file,
                        }
                    )
                elif surface == "tensor":
                    public_api = f"mindspore.Tensor.{api_name}"
                    rows.append(
                        {
                            "identity_key": f"{public_api}::{branch.op_branch}",
                            "public_api": public_api,
                            "public_surface": "mindspore.Tensor",
                            "api_name": api_name,
                            "op_branch": branch.op_branch,
                            "op": op_entry.op,
                            "primitive": op_entry.primitive,
                            "py_method": branch.py_method,
                            "interface": branch.interface,
                            "target_module": "mindspore.ops.api_def",
                            "target_symbol": api_name,
                            "resolver_kind": "api_def_interface",
                            "resolver_path": branch.source_file,
                            "source_file": branch.source_file,
                        }
                    )
    return rows


def build_mint_rows(
    ms_root: Path,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_branch: Dict[str, OpDefEntry],
    op_by_symbol: Dict[str, List[OpDefEntry]],
) -> List[dict]:
    mint_info = load_module_info(ms_root, "mindspore.mint")
    if mint_info is None:
        raise FileNotFoundError(f"Unable to parse mint exports from {MINT_INIT}")
    rows = []
    for public_name, binding in sorted(mint_info.from_imports.items()):
        if not binding.module.startswith("mindspore.ops"):
            continue
        resolved = resolve_symbol_to_entries(
            ms_root=ms_root,
            target_module=binding.module,
            target_symbol=binding.symbol or public_name,
            api_def_catalog=api_def_catalog,
            op_by_symbol=op_by_symbol,
        )
        public_api = f"mindspore.mint.{public_name}"
        for entry in resolved:
            op_entry = op_by_branch.get(entry["op_branch"])
            if op_entry is None:
                continue
            rows.append(
                {
                    "identity_key": f"{public_api}::{entry['op_branch']}",
                    "public_api": public_api,
                    "public_surface": "mindspore.mint",
                    "api_name": public_name,
                    "op_branch": entry["op_branch"],
                    "op": op_entry.op,
                    "primitive": op_entry.primitive,
                    "py_method": entry["py_method"],
                    "interface": entry["interface"],
                    "target_module": binding.module,
                    "target_symbol": binding.symbol or public_name,
                    "resolver_kind": entry["resolver_kind"],
                    "resolver_path": entry["resolver_path"],
                    "source_file": str(mint_info.path.relative_to(ms_root.parent)),
                }
            )
    return rows


def dedupe_rows(rows: Iterable[dict]) -> List[dict]:
    dedup: Dict[Tuple[str, str], dict] = {}
    for row in rows:
        key = (row["public_api"], row["op_branch"])
        current = dedup.get(key)
        if current is None:
            dedup[key] = row
            continue
        current_kind = current.get("resolver_kind", "")
        next_kind = row.get("resolver_kind", "")
        if current_kind == "op_def" and next_kind != "op_def":
            dedup[key] = row
    return sorted(dedup.values(), key=lambda item: (item["public_api"], item["op_branch"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build phase-1 api_identity index for operator-facts.")
    parser.add_argument("--ms-root", type=Path, default=DEFAULT_MS_ROOT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_DIR / "api_identity.jsonl")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_DIR / "api_identity.csv")
    args = parser.parse_args()

    op_by_branch, op_by_symbol = build_op_catalog(args.ms_root)
    api_def_catalog = build_api_def_catalog(args.ms_root)

    rows = []
    rows.extend(build_ops_and_tensor_rows(api_def_catalog, op_by_branch))
    rows.extend(build_mint_rows(args.ms_root, api_def_catalog, op_by_branch, op_by_symbol))
    rows = dedupe_rows(rows)

    write_jsonl(args.out_jsonl, rows)
    fieldnames = [
        "identity_key",
        "public_api",
        "public_surface",
        "api_name",
        "op_branch",
        "op",
        "primitive",
        "py_method",
        "interface",
        "target_module",
        "target_symbol",
        "resolver_kind",
        "resolver_path",
        "source_file",
    ]
    write_csv(args.out_csv, rows, fieldnames)
    print(f"api_identity_rows={len(rows)}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"out_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
