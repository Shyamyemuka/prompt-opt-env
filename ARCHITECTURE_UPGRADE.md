# LLM-Powered Actions Upgrade

## Problem with Current Implementation

Current actions are simple string manipulation:
- `add_context`: `prompt + "\nContext: " + task.context`
- `shorten`: regex-based filler removal
- `rephrase`: regex pattern matching

This is NOT intelligent prompt optimization. It's template injection.

## Solution: AI-Powered Actions

Each action now calls the LLM to actually transform the prompt intelligently.

### New Action Design

| Action | Old Behavior | New Behavior |
|--------|--------------|--------------|
| ADD_CONTEXT | Append canned context sentence | LLM rewrites prompt incorporating context naturally |
| SHORTEN | Regex remove filler words | LLM compresses prompt while preserving meaning |
| ADD_EXAMPLE | Append example format | LLM rewrites prompt to include examples organically |
| REPHRASE | Regex transformation | LLM rewrites for clarity and directness |
| ADD_CONSTRAINT | Append constraint sentence | LLM integrates constraints into prompt flow |
| REWRITE (NEW) | N/A | LLM completely rewrites prompt from scratch |

### Implementation Strategy

1. **Create `actions_llm.py`**: New module with LLM-powered transformations
2. **Keep backward compatibility**: Fall back to string actions if LLM fails
3. **Caching**: Cache LLM responses to avoid duplicate calls
4. **Cost control**: Track API calls, provide dry-run mode

### LLM Prompts for Each Action

```python
# ADD_CONTEXT
"""Given the task and current prompt, rewrite the prompt to naturally incorporate 
this context: {context}

Current prompt: {prompt}
Rewritten prompt:"""

# SHORTEN  
"""Compress this prompt to use fewer tokens while preserving all key information.
Remove filler words and redundancies.

Current prompt ({token_count} tokens): {prompt}
Compressed prompt:"""

# REPHRASE
"""Rewrite this prompt to be more direct, clear, and imperative.
Remove questions, politeness markers, and indirect phrasing.

Current prompt: {prompt}
Direct, clear prompt:"""

# REWRITE (full optimization)
"""You are a prompt optimization expert. Given a task and initial prompt,
rewrite the prompt to maximize clarity and specificity.

Task: {task_description}
Current prompt: {prompt}
Optimized prompt:"""
```

### Tradeoffs

**Pros:**
- Actually intelligent prompt optimization
- Judges will see real AI work
- Can demonstrate learning from feedback

**Cons:**
- More API calls (was 21 per benchmark, now ~100+)
- Slower execution
- Higher cost
- More failure modes

### Migration Plan

1. Create `actions_llm.py` with LLM-powered versions
2. Update `prompt_opt_env_environment.py` to use new actions
3. Update agent.py to leverage intelligent transformations
4. Update benchmark.py to track API calls
5. Test with limited episodes first
