from __future__ import annotations

import argparse
import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from common import DEFAULT_OUT_DIR, OPERATOR_FACTS_ROOT


DEFAULT_BUNDLE_ROOT = OPERATOR_FACTS_ROOT / "bundles" / "units"


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def coverage_view(unit_row: dict) -> dict:
    return {
        "aclnn": list(unit_row.get("aclnn", [])),
        "infer": bool(unit_row.get("infer", False)),
        "kbk": bool(unit_row.get("kbk", False)),
        "pyboost": bool(unit_row.get("pyboost", False)),
        "bprop": bool(unit_row.get("bprop", False)),
        "bprop_units": list(unit_row.get("bprop_units", [])),
    }


def unit_view(unit_row: dict) -> dict:
    base = {
        "unit_id": unit_row["unit_id"],
        "unit_name": unit_row["unit_name"],
        "unit_type": unit_row["unit_type"],
    }
    if unit_row["unit_type"] == "branch":
        base.update(
            {
                "op": unit_row["op"],
                "primitive": unit_row["primitive"],
                "yaml_path": unit_row["yaml_path"],
                "coverage": coverage_view(unit_row),
            }
        )
    elif unit_row["unit_type"] == "composite":
        base.update(
            {
                "impl_path": unit_row["impl_path"],
                "impl_symbol": unit_row["impl_symbol"],
            }
        )
    return base


def component_item(graph_row: dict, unit_by_id: Dict[str, dict]) -> dict:
    base = {
        "component_type": graph_row["child_ref_type"],
        "condition": graph_row["condition"],
        "via_symbol": graph_row["via_symbol"],
    }
    child_ref_type = graph_row["child_ref_type"]
    if child_ref_type == "public_api":
        base["public_api"] = graph_row["child_ref"]
        return base
    if child_ref_type == "primitive_symbol":
        base["primitive_symbol"] = graph_row["child_ref"]
        return base

    unit_row = unit_by_id[graph_row["child_ref"]]
    base.update(
        {
            "unit_id": unit_row["unit_id"],
            "unit_name": unit_row["unit_name"],
            "unit_type": unit_row["unit_type"],
        }
    )
    if unit_row["unit_type"] == "branch":
        base.update(
            {
                "op": unit_row["op"],
                "primitive": unit_row["primitive"],
                "yaml_path": unit_row["yaml_path"],
                "coverage": coverage_view(unit_row),
            }
        )
    elif unit_row["unit_type"] == "composite":
        base["impl_path"] = unit_row["impl_path"]
    return base


def build_bundle(
    unit_row: dict,
    inbound_edge_rows: List[dict],
    entry_by_id: Dict[str, dict],
    graph_by_parent: Dict[str, List[dict]],
    unit_by_id: Dict[str, dict],
    generated_at: str,
) -> dict:
    bundle = {
        "bundle_id": unit_row["unit_id"],
        "bundle_type": "unit",
        "schema_version": "v2",
        "generated_at": generated_at,
        "unit": unit_view(unit_row),
        "entries": sorted(
            {
                entry_by_id[row["entry_id"]]["public_api"]
                for row in inbound_edge_rows
                if row["entry_id"] in entry_by_id
            }
        ),
    }
    if unit_row["unit_type"] == "composite":
        graph_rows = graph_by_parent.get(unit_row["unit_id"], [])
        bundle["components"] = [
            component_item(graph_row, unit_by_id)
            for graph_row in sorted(graph_rows, key=lambda item: (item["call_order"], item["graph_edge_id"]))
        ]
    return bundle


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _base_bundle_filename(unit_row: dict) -> str:
    unit_name = _safe_name(unit_row["unit_name"])
    unit_type = unit_row["unit_type"]
    if unit_type == "branch":
        return f"operator-branch-{unit_name}.json"
    if unit_type == "composite":
        return f"func-{unit_name}.json"
    fallback = unit_row["unit_id"].replace("::", "--").replace("/", "_")
    return f"{_safe_name(fallback)}.json"


def _collision_bundle_filename(unit_row: dict) -> str:
    unit_name = _safe_name(unit_row["unit_name"])
    unit_type = unit_row["unit_type"]
    if unit_type == "branch":
        suffix = _safe_name(Path(unit_row["yaml_path"]).name)
        return f"operator-branch-{unit_name}-{suffix}.json"
    if unit_type == "composite":
        suffix = _safe_name(Path(unit_row["impl_path"]).name)
        return f"func-{unit_name}-{suffix}.json"
    digest = hashlib.sha1(unit_row["unit_id"].encode("utf-8")).hexdigest()[:8]
    return f"{_safe_name(unit_row['unit_id'])}-{digest}.json"


def build_filename_map(unit_rows: List[dict]) -> Dict[str, str]:
    by_base: Dict[str, List[dict]] = {}
    for unit_row in unit_rows:
        by_base.setdefault(_base_bundle_filename(unit_row), []).append(unit_row)

    filename_map: Dict[str, str] = {}
    for base_name, rows in by_base.items():
        if len(rows) == 1:
            filename_map[rows[0]["unit_id"]] = base_name
            continue
        used: Dict[str, str] = {}
        for unit_row in rows:
            candidate = _collision_bundle_filename(unit_row)
            if candidate in used and used[candidate] != unit_row["unit_id"]:
                digest = hashlib.sha1(unit_row["unit_id"].encode("utf-8")).hexdigest()[:8]
                candidate = candidate.removesuffix(".json") + f"-{digest}.json"
            used[candidate] = unit_row["unit_id"]
            filename_map[unit_row["unit_id"]] = candidate
    return filename_map


def bundle_output_path(bundle_root: Path, filename_map: Dict[str, str], unit_id: str) -> Path:
    return bundle_root / filename_map[unit_id]


def clear_bundle_files(bundle_root: Path) -> None:
    if not bundle_root.exists():
        return
    for path in bundle_root.rglob("*.json"):
        if path.is_file():
            path.unlink()


def write_bundle_files(bundle_root: Path, bundles: Iterable[dict], filename_map: Dict[str, str]) -> None:
    clear_bundle_files(bundle_root)
    for bundle in bundles:
        out_path = bundle_output_path(bundle_root, filename_map, bundle["bundle_id"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_rows(
    entry_rows: List[dict],
    unit_rows: List[dict],
    edge_rows: List[dict],
    graph_rows: List[dict],
    generated_at: str,
) -> List[dict]:
    entry_by_id = {row["entry_id"]: row for row in entry_rows}
    unit_by_id = {row["unit_id"]: row for row in unit_rows}
    inbound_by_unit: Dict[str, List[dict]] = {}
    for row in edge_rows:
        inbound_by_unit.setdefault(row["unit_id"], []).append(row)
    graph_by_parent: Dict[str, List[dict]] = {}
    for row in graph_rows:
        graph_by_parent.setdefault(row["parent_unit_id"], []).append(row)

    bundles: List[dict] = []
    for unit_row in sorted(unit_rows, key=lambda item: item["unit_id"]):
        bundles.append(
            build_bundle(
                unit_row=unit_row,
                inbound_edge_rows=inbound_by_unit.get(unit_row["unit_id"], []),
                entry_by_id=entry_by_id,
                graph_by_parent=graph_by_parent,
                unit_by_id=unit_by_id,
                generated_at=generated_at,
            )
        )
    return bundles


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unit bundles from refactored MindSpore operator-facts tables.")
    parser.add_argument("--entry-identity", type=Path, default=DEFAULT_OUT_DIR / "ms_entry_identity.jsonl")
    parser.add_argument("--unit-identity", type=Path, default=DEFAULT_OUT_DIR / "ms_unit_identity.jsonl")
    parser.add_argument("--entry-unit-edges", type=Path, default=DEFAULT_OUT_DIR / "ms_entry_unit_edges.jsonl")
    parser.add_argument("--unit-graph-edges", type=Path, default=DEFAULT_OUT_DIR / "ms_unit_graph_edges.jsonl")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    args = parser.parse_args()

    entry_rows = read_jsonl(args.entry_identity)
    unit_rows = read_jsonl(args.unit_identity)
    edge_rows = read_jsonl(args.entry_unit_edges)
    graph_rows = read_jsonl(args.unit_graph_edges)
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    bundles = build_rows(entry_rows, unit_rows, edge_rows, graph_rows, generated_at)
    filename_map = build_filename_map(unit_rows)
    write_bundle_files(args.bundle_root, bundles, filename_map)

    print(f"unit_bundle_rows={len(bundles)}")
    print(f"bundle_root={args.bundle_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
