# Reflexion Strategy

You MUST follow the Reflexion (Self-Correction) pattern throughout this task.

## Core Pattern

After EVERY major step, perform explicit self-assessment:

**Act** → **Self-Assess** → **Decide** (Continue or Refine)

## Required Behavior

After completing each significant action or phase:

### 1. Quality Check
- Did I achieve my objective?
- Is the work correct and complete?
- Are there any obvious issues?

### 2. Error Detection  
- Logic errors or bugs?
- Missing edge cases?
- Assumptions that might be wrong?
- Incomplete work?

### 3. Decision Point
- ✅ **Satisfactory**: Continue to next step
- ⚠️ **Issues found**: Refine and retry before proceeding

## Example Format

```
[Complete an action]

Self-Assessment:
✓ Implementation complete
✓ Follows project structure  
✗ Issue: Token expiration time hardcoded (should be configurable)
✗ Issue: No input validation on login endpoint

Refinement needed: Fix hardcoded value, add validation.

[Make fixes]

Self-Assessment (after refinement):
✓ Token expiration now configurable
✓ Input validation added
✓ All issues resolved

Proceeding to next step.
```

## Key Principles

- **Be honest in self-assessment** - don't rationalize away concerns
- **Iterate until satisfactory** - don't proceed with known issues
- **Document issues explicitly** - use ✓ and ✗ markers
- **Fix before moving on** - catch errors early

## Self-Assessment Checklist

After each major step, verify:
- [ ] Objective achieved?
- [ ] All requirements met?
- [ ] Edge cases considered?
- [ ] Error handling appropriate?
- [ ] Code quality acceptable?
