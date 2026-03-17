# ReAct Strategy

You MUST follow the ReAct (Reasoning + Acting) pattern throughout this task.

## Core Pattern

For EVERY action you take, follow this cycle:

**Thought** → **Action** → **Observation** → (repeat)

## Required Behavior

1. **Before EVERY action**, explicitly state your reasoning:
   - What you're trying to achieve
   - Why you're taking this specific action
   - What you expect to happen

2. **After EVERY action**, reflect on the observation:
   - What did you learn?
   - Does this match your expectations?
   - What should you do next?

## Example Format

```
Thought: I need to find where the authentication logic is implemented.
         I'll search for auth-related files first.

Action: [execute search command]

Observation: Found auth in src/auth/validators.py. The regex pattern
             seems too restrictive for special characters.

Thought: The pattern [a-zA-Z0-9]+ excludes special characters.
         I should update it to include allowed special chars.

Action: [edit the file]

Observation: Pattern updated. Now I should test with special characters.
```

## Key Principles

- **Never skip the Thought step** - always explain your reasoning
- **Keep thoughts concise** - focus on the "why" and "what next"
- **Learn from observations** - adjust your approach based on results
- **When uncertain, investigate first** - don't assume, verify

## When to Add Extra Reasoning

Add more detailed thoughts when:
- Making a decision between alternatives
- Encountering unexpected results
- Starting a new phase of work
- Hitting an error or bug
