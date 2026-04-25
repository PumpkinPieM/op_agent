# Temporary Operator Isolation For Fast Iteration

Use this workflow only when you need to validate **one newly added operator** quickly without paying for a full parameterized run on every iteration.

This is a **temporary debugging tactic**, not the default skill path and not a final repository state.

## What This Workflow Is For

Use it when all of the following are true:

- you are actively editing one new OpInfo registration
- the normal frontend parameterized test entry would run many unrelated operators
- you need a fast local or remote iteration loop before the final full validation

Do **not** treat this temporary isolation patch as part of the finished change. Remove it before final validation and before leaving the repo in a reviewable state.

## Minimal Flow

1. Make the real OpInfo change first.
2. Add a temporary override near the relevant `xxx_op_db` list so only the target operator remains in scope.
3. Run the targeted validation loop until the new operator path is stable.
4. Remove the temporary override patch.
5. Rerun the required full validation matrix.
6. Only keep the real OpInfo registration change; do not keep the isolation patch.

## Typical Shape Of The Temporary Patch

Example for `other_op_db`:

```python
other_op_db = [
    ...
]

# Temporary local isolation for fast iteration only.
other_op_db = ["mint.nn.Conv2d"]
```

For unrelated db lists, if needed for a specific local run, temporarily override them to `[]`.

## Rules

- Keep the patch minimal and easy to remove.
- Comment it clearly as temporary isolation.
- Never commit the isolation patch as part of the final clean change unless the user explicitly asks for it.
- After the targeted loop passes, restore the original db wiring and rerun the required full matrix.
- If remote validation is being used, apply the same tactic only in the temporary remote payload or temporary remote repo state, then restore it there as well.

## Recommended Use

Prefer this workflow for:

- fast local iteration on one operator
- temporary remote debugging when the full frontend matrix is too expensive

Do not use it to avoid final full validation. It is a speed tool, not a replacement for the normal completion criteria.
