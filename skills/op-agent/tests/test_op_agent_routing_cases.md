# op-agent Routing Cases

Prompt-eval style samples for the navigator skill. These are not executable
tests; they define the expected routing behavior and handoff discipline.
The machine-readable routing contract lives in `tests/routing_cases.yaml`.
This file is the human-readable companion, while exhaustive alias coverage
belongs in the YAML contract.
Here `target_backend_raw` shows the original user input before normalization.

## Case 1: CPU Default Dispatch (Plugin)

### Input
```text
api_name: mindspore.mint.abs
target_backend_raw: CPU
problem_type: operator-gap
known_evidence: The operator is missing on CPU. No specific integration requirement provided.
```

### Expected

- Normalize CPU as the target backend.
- Select `cpu-plugin-builder` as the best fit.
- Output a minimal handoff summary instead of a recommendation report.
- Load `cpu-plugin-builder` and start implementation immediately.

## Case 2: CPU Native Dispatch (Manual Override)

### Input

```text
api_name: mindspore.mint.xxx
target_backend_raw: CPU
problem_type: operator-gap
known_evidence: The user explicitly requires deep framework integration inside MindSpore core.
```

### Expected

- Recognize the explicit native requirement.
- Select `cpu-native-builder` as the best fit.
- Handoff directly to `cpu-native-builder`.
- Keep the navigator output at routing level only, with no code generation.

## Case 3: NPU Dispatch via Ascend Alias

### Input

```text
api_name: mindspore.mint.mul
target_backend_raw: Ascend
problem_type: operator-gap
known_evidence: This is an ACLNN-based implementation task.
```

### Expected

- Normalize "Ascend" to `NPU`.
- Select `aclnn-builder` as the best fit.
- Handoff directly to `aclnn-builder`.
- Start execution immediately after the routing decision.

## Case 4: GPU Direct Dispatch

### Input

```text
api_name: mindspore.mint.xxx
target_backend_raw: GPU
problem_type: operator-gap
known_evidence: The operator is unsupported on GPU.
```

### Expected

- Normalize GPU as the target backend.
- Select `gpu-builder` as the best fit.
- Do not describe GPU as roadmap-only.
- Handoff directly to `gpu-builder` and start implementation.

## Case 5: CPU Ambiguity Defaults to Plugin

### Input

```text
api_name: mindspore.ops.CustomOp
target_backend_raw: CPU
problem_type: operator-gap
known_evidence: Unsupported on CPU.
```

### Expected

- Recognize that no native-only requirement was provided.
- Default to `cpu-plugin-builder`.
- Avoid turning the navigator into a prolonged clarification step.
- Handoff immediately to `cpu-plugin-builder`.

## Case 6: mint API on NPU

### Input

```text
api_name: mindspore.mint.abs
target_backend_raw: NPU
problem_type: operator-gap
known_evidence: ""
```

### Expected

- Keep the normalized backend as `NPU`.
- Select `aclnn-builder` as the best fit.
- Handoff directly to `aclnn-builder`.
- Ensure the navigator still does not generate implementation code itself.
