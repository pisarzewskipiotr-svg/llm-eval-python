import sys
from pathlib import Path
from llm_eval.benchmarks.effibench import EffiBenchTask
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_eval.benchmarks.effibench import load_sample_50
from llm_eval.strategies import ZeroShotStrategy, ChainOfThoughtStrategy, SelfRefineStrategy
from llm_eval.strategies.base import StrategyState

tasks = load_sample_50()
task = tasks[0]  # Wildcard Matching

print("=" * 60)
print("=== ZERO-SHOT PROMPT ===")
print("=" * 60)
strategy = ZeroShotStrategy()
state = StrategyState(task=task)
prompt = strategy.build_prompt(state)
print(f"System: {prompt.system_message}")
print(f"\nUser:\n{prompt.user_message[:1500]}")
print(f"\n[Strategy: {prompt.strategy_name}, version: {prompt.template_version}, iter: {prompt.iteration}]")

print("\n" + "=" * 60)
print("=== CHAIN-OF-THOUGHT PROMPT ===")
print("=" * 60)
strategy = ChainOfThoughtStrategy()
state = StrategyState(task=task)
prompt = strategy.build_prompt(state)
print(f"User:\n{prompt.user_message[:1500]}")

print("\n" + "=" * 60)
print("=== SELF-REFINE INITIAL PROMPT ===")
print("=" * 60)
strategy = SelfRefineStrategy()
state = StrategyState(task=task, iteration=0)
prompt = strategy.build_prompt(state)
print(f"User:\n{prompt.user_message[:1500]}")

print("\n" + "=" * 60)
print("=== SELF-REFINE ITER-1 PROMPT (z fake feedbackiem) ===")
print("=" * 60)
fake_state = StrategyState(
    task=task,
    iteration=1,
    previous_code="class Solution:\n    def isMatch(self, s, p):\n        return False",
    previous_time_ms=5.234,
    canonical_time_ms=0.925,
    previous_functional_status="LOGICAL_REGRESSION"
)
prompt = strategy.build_prompt(fake_state)
print(f"User:\n{prompt.user_message[:2000]}")

print("\n" + "=" * 60)
print("=== TEST EKSTRAKCJI KODU ===")
print("=" * 60)

# Test 1 standard markdown
test1 = """Here's my solution:

```python
class Solution:
    def isMatch(self, s, p):
        return True
```

Done!"""
code, status = strategy.extract_code(test1)
print(f"Test 1 (standard markdown): status={status}")
print(f"Code: {repr(code)}")

# Test 2: niezamknięty blok (truncation)
test2 = """```python
class Solution:
    def isMatch(self, s, p):
        # Some logic
        return"""
code, status = strategy.extract_code(test2)
print(f"\nTest 2 (truncated): status={status}")
print(f"Code: {repr(code)}")

# Test 3: brak markdown
test3 = "class Solution:\n    def isMatch(self, s, p):\n        return True"
code, status = strategy.extract_code(test3)
print(f"\nTest 3 (no markdown): status={status}")
print(f"Code: {repr(code)}")
