"""Reviewed native-task data edits, guarded against upstream source drift."""

import json
from copy import deepcopy
from hashlib import sha256

from inspect_ai import Task

_MBPP_143_TESTS = [
    "assert find_lists(([1, 2, 3, 4], [5, 6, 7, 8])) == 2",
    "assert find_lists(([1, 2], [3, 4], [5, 6]))  == 3",
    "assert find_lists(([9, 8, 7, 6, 5, 4, 3, 2, 1],)) == 1",
]

_HARMONIC_PROMPT = (
    "Write a function harmonic_sum(n) returning 1 + 1/2 + ... + 1/n for a positive integer n."
)

PATCHES = {
    "humaneval": {
        "HumanEval/125": {
            "target": (
                "befd89f15b631cea2723b967affeb0ebef18c403ac385c4db9c2019e5343b10e",
                (
                    "    if any(c.isspace() for c in txt):\n"
                    "        return txt.split()\n"
                    '    elif "," in txt:\n'
                    "        return txt.replace(',',' ').split()\n"
                    "    else:\n"
                    "        return len([i for i in txt if i.islower() and ord(i)%2 == 0])\n"
                ),
            ),
            "metadata.test": (
                "947858152e43c24b16de7e21839f8df5865767f39826c9524478aab36a26fd60",
                (
                    "def check(candidate):\n"
                    "\n"
                    '    assert candidate("Hello world!") == ["Hello","world!"]\n'
                    '    assert candidate("Hello,world!") == ["Hello","world!"]\n'
                    '    assert candidate("Hello world,!") == ["Hello","world,!"]\n'
                    '    assert candidate("Hello,Hello,world !") == ["Hello,Hello,world","!"]\n'
                    '    assert candidate("abcdef") == 3\n'
                    '    assert candidate("aaabb") == 2\n'
                    '    assert candidate("aaaBb") == 1\n'
                    '    assert candidate("") == 0\n'
                    "\n"
                    '    for whitespace in "\\t\\n\\r\\v\\f":\n'
                    '        assert candidate("abc" + whitespace + "def") == ["abc", "def"]\n'
                ),
            ),
        },
    },
    "mbpp": {
        "143": {
            "target": (
                "fb6158add944a0c06d3ec55e0813a4dd3284af361476b2f017cc7ccca45b2153",
                _MBPP_143_TESTS,
            ),
            "metadata.test_list": (
                "fb6158add944a0c06d3ec55e0813a4dd3284af361476b2f017cc7ccca45b2153",
                _MBPP_143_TESTS,
            ),
            "metadata.test_list_str": (
                "2649e6b6975777a1b0afadda820216adb1313301cc39ec6019f9ba6dbef2dcfd",
                "\n".join(_MBPP_143_TESTS),
            ),
            "metadata.code": (
                "f3abaf3ccb02ec851c8b8b91287e352f6175440a114a604cefa07b4eb339795d",
                (
                    "def find_lists(items):\n"
                    "    return sum(isinstance(item, list) for item in items)\n"
                ),
            ),
        },
        "248": {
            "input": (
                "e7bd955d2086af5b81b25c576c6c5f7058f2720fee787d8a3183a16812ac9b31",
                _HARMONIC_PROMPT,
            ),
            "metadata.prompt": (
                "e7bd955d2086af5b81b25c576c6c5f7058f2720fee787d8a3183a16812ac9b31",
                _HARMONIC_PROMPT,
            ),
        },
    },
}


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def prepare_samples(task: Task, name: str):
    """Apply reviewed edits only when every original field still matches its hash."""
    for sample in task.dataset:
        changes = PATCHES.get(name, {}).get(str(sample.id), {})
        metadata = dict(sample.metadata or {})
        for field, (expected_digest, _) in changes.items():
            actual = (
                metadata.get(field[9:]) if field.startswith("metadata.") else getattr(sample, field)
            )
            if _digest(actual) != expected_digest:
                raise ValueError(f"{name}/{sample.id}: reviewed original changed for {field}")
        for field, (_, replacement) in changes.items():
            replacement = deepcopy(replacement)
            if field.startswith("metadata."):
                metadata[field[9:]] = replacement
            else:
                setattr(sample, field, replacement)
        if changes:
            sample.metadata = metadata
