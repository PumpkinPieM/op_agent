# remote_deploy_and_test Implementation Notes (for skill development only)

<a id="remote-deploy-dev-doc-positioning"></a>
## 1. Document Positioning

1. This document describes the implementation constraints and output contract of the remote runner.
2. This document is not used as skill execution input. During execution, only `workflow/remote_deploy_and_test.md` is used.
3. The output specifications in this document are intended for future script evolution.
4. The design does not need to overemphasize compatibility concerns.

<a id="remote-deploy-dev-artifact-layout"></a>
## 2. Job Artifact Directory

1. The artifact directory for each job is fixed as `<artifact_root>/<job_id>/`.
2. The default `artifact_root` is `/tmp/op_info_artifacts`.
3. The outputs in the directory include `pytest.log`, `junit.xml`, `env.txt`, `deploy_meta.json`, and `summary.json`.

<a id="remote-deploy-dev-artifact-overview"></a>
## 3. Output Overview

| Output | Format | Complexity | Generation Condition |
| --- | --- | --- | --- |
| `pytest.log` | UTF-8 text | Medium | Always generated |
| `junit.xml` | XML (pytest junit) | Medium | When `test_cmd` uses pytest and writes to the default path |
| `env.txt` | UTF-8 text (`key=value`) | Low | Always generated |
| `deploy_meta.json` | JSON | Medium | Always generated |
| `summary.json` | JSON | High | Always generated |

Notes:

1. Outputs with `Complexity = Medium/High` require templates located under the `template/` directory.
2. All JSON outputs are written by the server-side `write_json` helper using UTF-8, `ensure_ascii=true`, two-space indentation, and sorted keys.

<a id="remote-deploy-dev-artifact-spec"></a>
## 4. Detailed Specification for Each Output

<a id="remote-deploy-dev-pytest-log"></a>
### 4.1 `pytest.log`

Content requirements:

1. Each line must be prefixed with a timestamp in the format `[YYYY-MM-DD HH:MM:SS]`.
2. It must include at least one structured record: `[runner] job_id=...`.
3. It must include at least one structured record: `[runner] payload=...` (JSON string).
4. Each command must include at least `[exec] cwd=...`, `[exec] cmd=...`, and `[exec] return_code=...`.
5. Command stdout and stderr must be merged into the same log file and appended in read order.

Format requirements:

1. UTF-8 text.
2. Newline character `\n`.
3. Unstructured business output is allowed, but it must not break the structured records above.

Template:

1. `template/pytest_log.template.txt`

<a id="remote-deploy-dev-junit-xml"></a>
### 4.2 `junit.xml`

Content requirements:

1. It must be junit XML that pytest can parse.
2. Failed test cases must appear under `testcase/failure` or `testcase/error`.
3. The `classname` and `name` fields must combine into a test-case identifier (`classname::name`).

Generation requirements:

1. When `test_cmd` contains `pytest` and `--junitxml` is not explicitly set, the server automatically appends:
   `--junitxml=<artifact_dir>/junit.xml`.
2. If the user already set `--junitxml` in `test_cmd`, the server must not override it.

Template:

1. `template/junit_xml.template.xml`

<a id="remote-deploy-dev-env-txt"></a>
### 4.3 `env.txt`

Content requirements:

1. Each line must be `key=value`.
2. The minimum required field set must include `time`.
3. The minimum required field set must include `host`.
4. The minimum required field set must include `python`.
5. The minimum required field set must include `git`.

Format requirements:

1. UTF-8 text.
2. Preserve a trailing newline at the end of the file.

<a id="remote-deploy-dev-deploy-meta"></a>
### 4.4 `deploy_meta.json`

Fields and requirements:

1. `branch` (string, required): the branch used by the job.
2. `commit` (string, required, may be empty): the commit specified for the job; use an empty string if not specified.
3. `deploy_time` (string, required): UTC ISO8601 time, for example `2026-03-04T03:15:20+00:00`.
4. `runner_id` (string, required): the server hostname (`os.uname().nodename`).

Template:

1. `template/deploy_meta.template.json`

<a id="remote-deploy-dev-summary-json"></a>
### 4.5 `summary.json`

Fields and requirements:

1. `job_id` (string, required).
2. `status` (string, required): `success`, `failed`, `timeout`, or `canceled`.
3. `failed_cases` (array[string], required): the list of failed test cases, deduplicated while preserving original order.
4. `error_type` (string, required): `""`, `infra`, or `testcase`.
5. `top_traceback` (string, required): the key error snippet; it must be an empty string when `status=success`.
6. `generated_at` (string, required): UTC ISO8601 time.

Status consistency requirements:

1. When a single job has `status=success`, `failed_cases=[]`, `error_type=""`, and `top_traceback=""`.
2. When `status=failed` because of a test failure, `error_type=testcase`.
3. When `status=failed` because of environment, deployment, dependency, or similar issues, `error_type=infra`.
4. When `status=timeout` or `status=canceled`, `error_type=infra`.

Additional end-to-end workflow requirements:

1. The runner keeps single-job semantics and must not implicitly chain a second test on the server side.
2. After the first-round functional test returns `status=success`, the skill workflow layer must append `--count=50` to the same `payload.test_cmd` and submit a separate stability job.
3. Overall validation is complete only if the stability job also returns `status=success`.

Template:

1. `template/summary.template.json`

<a id="remote-deploy-dev-status-api"></a>
## 5. API Output Specification (Status Query)

`GET /jobs/{job_id}` returns the job status payload. Its structure is used by the client for polling and end-to-end workflow decisions.

Field requirements:

1. `job_id` (string, required).
2. `status` (string, required).
3. `payload` (object, required): a normalized copy of the original submission parameters.
4. `artifact_uri` (string, required): `/artifacts/<job_id>/`.
5. `artifact_bundle_uri` (string, required): `/jobs/<job_id>/artifacts.zip`.
6. `error_type` (string, required).
7. `created_at`, `started_at`, and `finished_at` (string/null, required).
8. `api_version` (string, required).
9. `summary` (object, required for terminal states): same structure as `summary.json`.

Template:

1. `template/job_status_response.template.json`

<a id="remote-deploy-dev-download-api"></a>
## 6. API Output Specification (Artifact Download)

`GET /jobs/{job_id}/artifacts.zip` returns a zip archive containing all artifacts for the current job, allowing the client to download them for analysis after `status=failed`.

Behavior requirements:

1. Return 404 if `job_id` does not exist.
2. Return 404 if the artifact directory does not exist.
3. Return `Content-Type: application/zip`.
4. Return `Content-Disposition: attachment; filename="<job_id>_artifacts.zip"`.

<a id="remote-deploy-dev-templates"></a>
## 7. Template List

1. `template/summary.template.json`
2. `template/deploy_meta.template.json`
3. `template/job_status_response.template.json`
4. `template/pytest_log.template.txt`
5. `template/junit_xml.template.xml`
