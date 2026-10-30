"""Reviewed data edits for pinned public samples, checked before preparation."""

from copy import deepcopy

from inspect_ai import Task

HUMANEVAL_GRADER_VERSION = 1

PATCHES = {
    "humaneval": {
        "HumanEval/125": {
            "target": (
                '    if " " in txt:\n'
                "        return txt.split()\n"
                '    elif "," in txt:\n'
                "        return txt.replace(',',' ').split()\n"
                "    else:\n"
                "        return len([i for i in txt if i.islower() "
                "and ord(i)%2 == 0])\n",
                "    if any(c.isspace() for c in txt):\n"
                "        return txt.split()\n"
                '    elif "," in txt:\n'
                "        return txt.replace(',',' ').split()\n"
                "    else:\n"
                "        return len([i for i in txt if i.islower() "
                "and ord(i)%2 == 0])\n",
            ),
            "metadata.test": (
                "def check(candidate):\n"
                "\n"
                '    assert candidate("Hello world!") == '
                '["Hello","world!"]\n'
                '    assert candidate("Hello,world!") == '
                '["Hello","world!"]\n'
                '    assert candidate("Hello world,!") == '
                '["Hello","world,!"]\n'
                '    assert candidate("Hello,Hello,world !") '
                '== ["Hello,Hello,world","!"]\n'
                '    assert candidate("abcdef") == 3\n'
                '    assert candidate("aaabb") == 2\n'
                '    assert candidate("aaaBb") == 1\n'
                '    assert candidate("") == 0\n',
                "def check(candidate):\n"
                "\n"
                '    assert candidate("Hello world!") == '
                '["Hello","world!"]\n'
                '    assert candidate("Hello,world!") == '
                '["Hello","world!"]\n'
                '    assert candidate("Hello world,!") == '
                '["Hello","world,!"]\n'
                '    assert candidate("Hello,Hello,world !") '
                '== ["Hello,Hello,world","!"]\n'
                '    assert candidate("abcdef") == 3\n'
                '    assert candidate("aaabb") == 2\n'
                '    assert candidate("aaaBb") == 1\n'
                '    assert candidate("") == 0\n'
                "\n"
                '    for whitespace in "\\t\\n\\r\\v\\f":\n'
                '        assert candidate("abc" + whitespace '
                '+ "def") == ["abc", "def"]\n',
            ),
        }
    },
    "mbpp": {
        "143": {
            "target": (
                [
                    "assert find_lists(([1, 2, 3, 4], [5, 6, 7, 8])) == 2",
                    "assert find_lists(([1, 2], [3, 4], [5, 6]))  == 3",
                    "assert find_lists(([9, 8, 7, 6, 5, 4, 3, 2, 1])) == 1",
                ],
                [
                    "assert find_lists(([1, 2, 3, 4], [5, 6, 7, 8])) == 2",
                    "assert find_lists(([1, 2], [3, 4], [5, 6]))  == 3",
                    "assert find_lists(([9, 8, 7, 6, 5, 4, 3, 2, 1],)) == 1",
                ],
            ),
            "metadata.test_list": (
                [
                    "assert find_lists(([1, 2, 3, 4], [5, 6, 7, 8])) == 2",
                    "assert find_lists(([1, 2], [3, 4], [5, 6]))  == 3",
                    "assert find_lists(([9, 8, 7, 6, 5, 4, 3, 2, 1])) == 1",
                ],
                [
                    "assert find_lists(([1, 2, 3, 4], [5, 6, 7, 8])) == 2",
                    "assert find_lists(([1, 2], [3, 4], [5, 6]))  == 3",
                    "assert find_lists(([9, 8, 7, 6, 5, 4, 3, 2, 1],)) == 1",
                ],
            ),
            "metadata.test_list_str": (
                "assert find_lists(([1, 2, 3, 4], [5, 6, 7, 8])) == "
                "2\n"
                "assert find_lists(([1, 2], [3, 4], [5, 6]))  == 3\n"
                "assert find_lists(([9, 8, 7, 6, 5, 4, 3, 2, 1])) "
                "== 1",
                "assert find_lists(([1, 2, 3, 4], [5, 6, 7, 8])) == "
                "2\n"
                "assert find_lists(([1, 2], [3, 4], [5, 6]))  == 3\n"
                "assert find_lists(([9, 8, 7, 6, 5, 4, 3, 2, 1],)) "
                "== 1",
            ),
            "metadata.code": (
                "def find_lists(Input): \n"
                "\tif isinstance(Input, list): \n"
                "\t\treturn 1\n"
                "\telse: \n"
                "\t\treturn len(Input) ",
                "def find_lists(items):\n"
                "    return sum(isinstance(item, list) for item in "
                "items)\n",
            ),
        },
        "248": {
            "input": (
                "Write a function that takes in an integer n and calculates the "
                "harmonic sum of n-1.",
                "Write a function harmonic_sum(n) returning 1 + 1/2 + ... + 1/n for "
                "a positive integer n.",
            ),
            "metadata.prompt": (
                "Write a function that takes in an integer n and "
                "calculates the harmonic sum of n-1.",
                "Write a function harmonic_sum(n) returning 1 + 1/2 + ... "
                "+ 1/n for a positive integer n.",
            ),
        },
    },
    "math": {
        "math_98fed519": {
            "target": ("1,-1,-3i", "1,-1,-3i"),
            "metadata.math_answer_order": (None, "unordered"),
        },
        "math_b2cca4da": {
            "target": ("4", "4"),
            "metadata.math_answer_units": (None, "inches per second"),
        },
        "math_cb449b2b": {"target": ("100", "100"), "metadata.math_answer_units": (None, "steps")},
    },
}


def prepare_samples(task: Task, name: str):
    """Apply reviewed edits only when every original field still matches."""
    for sample in task.dataset:
        changes = PATCHES.get(name, {}).get(str(sample.id), {})
        metadata = dict(sample.metadata or {})
        for field, (expected, _) in changes.items():
            actual = (
                metadata.get(field[9:]) if field.startswith("metadata.") else getattr(sample, field)
            )
            if actual != expected:
                raise ValueError(f"{name}/{sample.id}: reviewed original changed for {field}")
        for field, (_, replacement) in changes.items():
            replacement = deepcopy(replacement)
            if field.startswith("metadata."):
                metadata[field[9:]] = replacement
            else:
                setattr(sample, field, replacement)
        if changes:
            sample.metadata = metadata
