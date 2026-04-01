from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from common import DEFAULT_OUT_DIR, REPO_ROOT, write_jsonl


DEFAULT_BUNDLE_ROOT = REPO_ROOT / "operator-facts" / "bundles"


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


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def parse_json_array(value: str) -> List[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if str(item)]


def parse_csv_list(value: str) -> List[str]:
    return [item for item in (part.strip() for part in value.split(",")) if item]


def bundle_output_path(bundle_root: Path, public_api: str, op_branch: str) -> Path:
    return bundle_root / public_api / f"{op_branch}.json"


def build_bundle(identity_row: dict, coverage_row: dict, generated_at: str) -> dict:
    aclnn = parse_json_array(coverage_row.get("aclnn", ""))
    aclnn_source = parse_csv_list(coverage_row.get("aclnn_source", ""))
    evidence = {
        "infer": parse_json_array(coverage_row.get("infer_evidence", "")),
        "pyboost": parse_json_array(coverage_row.get("pyboost_evidence", "")),
        "kbk": parse_json_array(coverage_row.get("kbk_evidence", "")),
        "bprop": parse_json_array(coverage_row.get("bprop_evidence", "")),
        "ut": parse_json_array(coverage_row.get("ut_evidence", "")),
        "st": parse_json_array(coverage_row.get("st_evidence", "")),
        "docs_cn": parse_json_array(coverage_row.get("docs_cn_evidence", "")),
        "docs_en": parse_json_array(coverage_row.get("docs_en_evidence", "")),
    }
    bundle = {
        "bundle_key": identity_row["identity_key"],
        "schema_version": "v1",
        "generated_at": generated_at,
        "identity": {
            "public_api": identity_row["public_api"],
            "public_surface": identity_row["public_surface"],
            "api_name": identity_row.get("api_name", ""),
            "op_branch": identity_row["op_branch"],
            "op": identity_row["op"],
            "primitive": identity_row["primitive"],
        },
        "resolver": {
            "py_method": identity_row.get("py_method", ""),
            "interface": identity_row.get("interface", ""),
            "target_module": identity_row.get("target_module", ""),
            "target_symbol": identity_row.get("target_symbol", ""),
            "resolver_kind": identity_row.get("resolver_kind", ""),
            "resolver_path": identity_row.get("resolver_path", ""),
            "source_file": identity_row.get("source_file", ""),
        },
        "coverage": {
            "coverage_key": coverage_row["coverage_key"],
            "op_yaml_path": coverage_row.get("op_yaml_path", ""),
            "class_name": coverage_row.get("class_name", ""),
            "dispatch_enable": parse_bool(coverage_row.get("dispatch_enable", "")),
            "dispatch_kind": coverage_row.get("dispatch_kind", ""),
            "dispatch_ascend": coverage_row.get("dispatch_ascend", ""),
            "aclnn": aclnn,
            "aclnn_source": aclnn_source,
            "infer": parse_bool(coverage_row.get("infer", "")),
            "pyboost": parse_bool(coverage_row.get("pyboost", "")),
            "kbk": parse_bool(coverage_row.get("kbk", "")),
            "bprop": parse_bool(coverage_row.get("bprop", "")),
            "ut": parse_bool(coverage_row.get("ut", "")),
            "st": parse_bool(coverage_row.get("st", "")),
            "docs_cn": parse_bool(coverage_row.get("docs_cn", "")),
            "docs_en": parse_bool(coverage_row.get("docs_en", "")),
        },
        "evidence": evidence,
        "refs": {
            "identity_key": identity_row["identity_key"],
            "coverage_key": coverage_row["coverage_key"],
        },
    }
    return bundle


def write_bundle_files(bundle_root: Path, bundles: Iterable[dict]) -> None:
    for bundle in bundles:
        identity = bundle["identity"]
        out_path = bundle_output_path(bundle_root, identity["public_api"], identity["op_branch"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build phase-1 op bundles for operator-facts.")
    parser.add_argument("--api-identity", type=Path, default=DEFAULT_OUT_DIR / "api_identity.jsonl")
    parser.add_argument("--ms-coverage", type=Path, default=DEFAULT_OUT_DIR / "ms_coverage.jsonl")
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_DIR / "op_bundles.jsonl")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    args = parser.parse_args()

    identity_rows = read_jsonl(args.api_identity)
    coverage_rows = read_jsonl(args.ms_coverage)
    coverage_by_key: Dict[str, dict] = {
        f"{row['op']}::{row['primitive']}": row for row in coverage_rows
    }
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    bundles: List[dict] = []
    missing_coverage: List[str] = []
    for identity_row in identity_rows:
        coverage_key = f"{identity_row['op']}::{identity_row['primitive']}"
        coverage_row = coverage_by_key.get(coverage_key)
        if coverage_row is None:
            missing_coverage.append(identity_row["identity_key"])
            continue
        bundles.append(build_bundle(identity_row, coverage_row, generated_at))

    bundles.sort(key=lambda item: item["bundle_key"])
    write_jsonl(args.out_jsonl, bundles)
    write_bundle_files(args.bundle_root, bundles)

    print(f"op_bundle_rows={len(bundles)}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"bundle_root={args.bundle_root}")
    if missing_coverage:
        print(f"missing_coverage={len(missing_coverage)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
