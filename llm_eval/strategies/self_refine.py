"""Self-Refine strategia: iteracyjne ulepszanie z feedbackiem wydajności."""

from llm_eval.strategies.base import PromptStrategy, PromptResult, StrategyState


SYSTEM_MESSAGE = "You are an expert Python programmer specializing in writing efficient, optimized code."

INITIAL_PROMPT = """Solve the following LeetCode problem with the most efficient solution possible.

PROBLEM:
{description}

REQUIREMENTS:
- Implement a class named `Solution` with the required method.
- Provide ONLY the Python code inside a ```python``` code block.

Solution:"""

REFINE_PROMPT = """You previously generated this solution:

```python
{previous_code}
```

PERFORMANCE FEEDBACK:
- Your solution executes in {previous_time_ms:.3f} ms (median over 15 runs).
- The canonical SOTA solution executes in {canonical_time_ms:.3f} ms.
- Your solution is {ratio:.2f}x {comparison} than the canonical baseline.
- Functional correctness: {functional_status}.

REFINE the solution to improve efficiency. Consider:
1. **Algorithmic complexity**: Can you reduce time complexity (e.g., O(n²) → O(n log n))?
2. **Data structures**: Are you using the most efficient structures (set/dict for lookup, heap for priority)?
3. **Redundant computations**: Eliminate repeated work, use memoization where appropriate.
4. **Python idioms**: Use list comprehensions, built-ins, `bisect`, `itertools` for compact and fast code.

**CRITICAL**: Maintain functional correctness — the solution MUST still pass all tests.

Provide the REFINED code inside a ```python``` code block.

Refined solution:"""


class SelfRefineStrategy(PromptStrategy):
    strategy_name = "self_refine"
    template_version = "v1"
    max_iterations = 2  # zgodnie z ustaleniem
    
    def build_prompt(self, state: StrategyState) -> PromptResult:
        if state.iteration == 0:
            # Pierwsza iteracja — taki sam prompt jak zero-shot
            user_msg = INITIAL_PROMPT.format(
                description=state.task.markdown_description.strip()
            )
        else:
            # Iteracja refinement — feedback o wydajności
            if state.previous_code is None or state.previous_time_ms is None:
                raise ValueError(
                    f"SelfRefine iteration {state.iteration} requires previous_code "
                    f"and previous_time_ms in state."
                )
            
            ratio = state.canonical_time_ms / state.previous_time_ms
            comparison = "slower" if ratio < 1 else "faster"
            
            user_msg = REFINE_PROMPT.format(
                previous_code=state.previous_code,
                previous_time_ms=state.previous_time_ms,
                canonical_time_ms=state.canonical_time_ms,
                ratio=1/ratio if ratio < 1 else ratio,
                comparison=comparison,
                functional_status=state.previous_functional_status or "passed"
            )
        
        return PromptResult(
            system_message=SYSTEM_MESSAGE,
            user_message=user_msg,
            strategy_name=self.strategy_name,
            template_version=self.template_version,
            iteration=state.iteration
        )