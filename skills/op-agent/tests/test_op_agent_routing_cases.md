# op-agent Routing Cases

Prompt-eval style samples for the navigator skill. These are not executable
tests; they define the expected routing behavior and wording discipline.
The machine-readable routing contract lives in `tests/routing_cases.yaml`.
This file is the human-readable companion, while exhaustive alias coverage
belongs in the YAML contract.
Here `target_backend_raw` shows the original user input before normalization.

## Case 1: CPU Recommended Path (Plugin)

### Input
```text
api_name: mindspore.mint.abs
target_backend_raw: CPU
problem_type: operator-gap
known_evidence: The operator is missing on CPU. No specific integration requirement provided.
```
  
### Expected

- Maturity Alignment: Identify that for CPU, the Plugin path is the Mature/Recommended option.
- Explanation: Briefly explain Native vs. Plugin but highlight the maturity of the CPU Plugin.
- Route: Recommend cpu-plugin-builder.
- Justification: State that the plugin path is currently the most mature and recommended workflow for CPU operator expansion.

## Case 2: CPU Native Path (Manual Override)

### Input

```text
api_name: mindspore.mint.xxx
target_backend_raw: CPU
problem_type: operator-gap
known_evidence: The user explicitly requires deep framework integration inside MindSpore core.
```

### Expected

- Requirement Recognition: Acknowledge the user's specific requirement for a Native implementation.
- Route: route to `cpu-native-builder`
- Status Note: Mention that while the Plugin path is generally recommended for CPU, the Native path is available and utilized for core integration needs.

## Case 3: NPU Standard Path (ACLNN)

### Input

```text
api_name: mindspore.mint.mul
target_backend_raw: Ascend
problem_type: operator-gap
known_evidence: This is an ACLNN-based implementation task.
```

### Expected

- Terminology Mapping: Map "Ascend" to the "NPU" target backend.
- Maturity Alignment: Identify npu-native-builder as the Mature/Standard path for NPU.
- Rule Application: Apply the rule that ACLNN adaptations belong exclusively to the NPU Native workflow.
- Route: Route to npu-native-builder.

## Case 4: GPU Gaps (Roadmap/Planned)

### Input

```text
api_name: mindspore.mint.xxx
target_backend_raw: GPU
problem_type: operator-gap
known_evidence: ""
```

### Expected

- Maturity Alignment: Identify that both GPU builders are currently in the Planned/Roadmap phase.
- Route: Set "Best fit" to Roadmap.
- Next Step: Inform the user that GPU support is on the roadmap and provide instructions for tracking its availability.

## Case 5: CPU Ambiguity Handling

### Input

```text
api_name: mindspore.ops.CustomOp
target_backend_raw: CPU
problem_type: operator-gap
known_evidence: Unsupported on CPU.
```

### Expected

- Identify Options: Present both cpu-plugin-builder and cpu-native-builder.
- Proactive Recommendation: Explicitly recommend the Plugin path first due to its higher maturity.
- Decision Signal: Ask the user if there are specific architectural reasons to choose the Native path over the recommended Plugin path.

## Case 6: Mint API on NPU

### Input

```text
api_name: mindspore.mint.abs
target_backend_raw: NPU
problem_type: operator-gap
known_evidence: ""
```

### Expected

- Namespace Logic: Recognize that the mint namespace strongly implies the Native implementation path for NPU.
- Maturity Alignment: Route to npu-native-builder as the established standard for NPU/Ascend.
- Constraint Adherence: Ensure no implementation logic or kernel code is generated in the response.
