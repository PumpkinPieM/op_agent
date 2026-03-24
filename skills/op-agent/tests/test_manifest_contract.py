import json
from pathlib import Path

import jsonschema
import yaml


def test_manifest_contract():
    skill_root = Path(__file__).resolve().parents[1]
    skill_md = skill_root / "SKILL.md"
    manifest = skill_root / "skill.yaml"
    schema = skill_root.parent / "_shared" / "contract" / "skill.schema.json"

    assert skill_md.exists()
    assert manifest.exists()
    assert schema.exists()

    manifest_data = yaml.safe_load(manifest.read_text())
    schema_data = json.loads(schema.read_text())
    jsonschema.validate(instance=manifest_data, schema=schema_data)

    assert manifest_data["entry"]["type"] == "manual"
    assert manifest_data["entry"]["path"] == "SKILL.md"
