# Plan-and-Execute Strategy

You MUST follow the Plan-and-Execute pattern throughout this task.

## Core Pattern

**Phase 1: Planning** → **Phase 2: Execution** → **Phase 3: Re-planning**

## Required Behavior

### Phase 1: Planning (BEFORE starting work)

Create an explicit numbered plan:

1. **Analyze the complete task** - break it into subtasks
2. **Identify dependencies** - what must happen before what
3. **Document your plan** - write it out explicitly

Example:
```
PLAN:
1. Research existing patterns in codebase [no deps]
2. Set up authentication middleware [depends on 1]
3. Implement password hashing [depends on 1]
4. Create login route [depends on 2, 3]
5. Create logout route [depends on 2]
6. Write tests [depends on 4, 5]
```

### Phase 2: Execution

For each subtask:
1. State which subtask you're executing
2. Complete the work
3. Mark the subtask complete
4. Verify dependencies for next subtask

Example:
```
Executing Subtask 2: Set up auth middleware
[work...]
Subtask 2 COMPLETE ✓

Subtask 4 depends on 2 and 3.
Subtask 3 status: not started
Starting Subtask 3 next.
```

### Phase 3: Re-planning

Update your plan when you discover:
- A subtask is no longer needed
- Additional subtasks are required
- Dependencies have changed
- A better approach exists

Example:
```
PLAN UPDATE:
- Adding Subtask 7: Rate limiting (discovered during implementation)
- Cancelling Subtask 5: logout handled by middleware
- New order: 1 → 2 → 3 → 7 → 4 → 6
```

## Key Principles

- **Plan upfront** - don't start work without a plan
- **Track progress explicitly** - mark subtasks complete
- **Re-plan when needed** - plans are living documents
- **Verify dependencies** - don't start a task before dependencies are done
- **Balance depth** - start high-level, add detail as you learn
