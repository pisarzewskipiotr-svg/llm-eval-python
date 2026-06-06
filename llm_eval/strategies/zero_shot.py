"""Zero-shot strategia: czysty prompt bez przykładów ani rozumowania."""

from llm_eval.strategies.base import PromptStrategy, PromptResult, StrategyState


SYSTEM_MESSAGE = "You are an expert Python programmer specializing in writing efficient, optimized code."

PROMPT_TEMPLATE = """Solve the following LeetCode problem with the most efficient solution possible.
Optimize for execution time and memory usage.

PROBLEM:
{description}

REQUIREMENTS:
- Implement a class named `Solution` with the required method.
- Optimize for time and space complexity (use efficient data structures and algorithms).
- Provide ONLY the Python code inside a ```python``` code block.
- Do NOT include explanations, tests, or example usage in the code block.

Solution:"""


class ZeroShotStrategy(PromptStrategy):
    strategy_name = "zero_shot"
    template_version = "v1"
    
    def build_prompt(self, state: StrategyState) -> PromptResult:
        # Dla zero-shot zawsze iteration=0
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