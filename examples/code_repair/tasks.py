"""Task 'code.python': write a Python function that passes the given unit tests (MBPP, Austin et al. 2021, CC BY 4.0,
data/mbpp.jsonl). Standard splits by task id: prompt 1-10, test 11-510, validation 511-600, train 601-974.

Producers run the train split (their own traffic); the validator holds out the test split. Nothing from the test split is
ever trained on.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TASK, CHECKER = "code.python", "mbpp-tests@1"
SPLITS = {"prompt": range(1, 11), "test": range(11, 511), "validation": range(511, 601), "train": range(601, 975)}


def load(split):
    ids = set(SPLITS[split])
    with open(os.path.join(HERE, "data", "mbpp.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if r["task_id"] in ids]


def prompt(task):
    """The standard MBPP prompt: the description plus the tests, which also fix the function's name and signature."""
    tests = "\n".join(task["test_list"])
    return (f"You are an expert Python programmer, and here is your task: {task['text']}\n"
            f"Your code should pass these tests:\n\n{tests}\n\n"
            "Write the complete function in a single ```python code block.")


def repair_prompt(task, code, feedback):
    """Send the failing code back with the checker's traceback: the 'traceback' in traceX, literally."""
    return (prompt(task) + "\n\nYour previous attempt:\n```python\n" + code.strip() + "\n```\n"
            f"failed the tests:\n{feedback}\n\nFix the code. Reply with the complete corrected function in a single "
            "```python code block.")
