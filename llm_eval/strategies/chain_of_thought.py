"""Chain-of-Thought strategia: jawne rozumowanie krok po kroku przed kodowaniem."""

from llm_eval.strategies.base import PromptStrategy, PromptResult, StrategyState


SYSTEM_MESSAGE = "You are an expert Python programmer specializing in writing efficient, optimized code."

PROMPT_TEMPLATE = """Solve the following LeetCode problem with maximum efficiency.

PROBLEM:
{description}

THINK STEP BY STEP before writing the code:

1. **Analyze the problem**:
   - What is the input? What is the output?
   - What are the constraints?
   - What is the time complexity of a brute-force approach?

2. **Identify the optimal approach**:
   - What data structure minimizes time complexity?
   - What algorithm has the best asymptotic complexity for this problem?
   - Can we trade space for time with memoization?

3. **Analyze complexity**:
   - Time complexity of your approach: O(?)
   - Space complexity: O(?)

4. **Implement**:
   - **MANDATORY: The code MUST be a class named `Solution` with the required method.**
   - The required method signature is the one matching the problem (e.g., `def isMatch(self, s, p)`).
   - Efficient data structures (sets, dicts, deques where appropriate).
   - Avoid redundant computations.

**REQUIRED OUTPUT FORMAT** (the code block MUST start with `class Solution`):

```python
class Solution:
    def methodName(self, ...):
        # your implementation
        return result
```

After your reasoning, provide the FINAL CODE inside a ```python``` code block.
- Code MUST start with `class Solution:` 
- Do NOT use standalone functions outside the class
- Do NOT include print statements, example usage, or tests
- Output ONLY the Solution class

Reasoning and Solution:"""


class ChainOfThoughtStrategy(PromptStrategy):
    strategy_name = "cot"
    template_version = "v2"  # ← zmiana z v1
    
    def build_prompt(self, state: StrategyState) -> PromptResult:
        user_msg = PROMPT_TEMPLATE.format(
            description=state.task.markdown_description.strip()
        )
        return PromptResult(
            system_message=SYSTEM_MESSAGE,
            user_message=user_msg,
            strategy_name=self.strategy_name,
            template_version=self.template_version,
            iteration=0
        )