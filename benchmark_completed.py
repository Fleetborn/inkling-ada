"""Compare completed Inkling answers over the loopback OpenAI API."""

import argparse
from hashlib import sha256
import json

from benchmark import generate


def cases(run):
    prefix = f"Exercise {run}. "
    yield (
        "arithmetic", prefix + "Reply with only the integer result of 17 times 19.", "323",
    )
    yield (
        "prose",
        prefix + "Write a three-sentence memo explaining why a stock review should record "
        "lead time and a reorder threshold. Use the phrases 'lead time' and 'reorder threshold'.",
        None,
    )
    for size in (105, 420):
        first, second = size // 3, size - 4
        inventory = "\n".join(
            f"Product {i:04d}: stock {(i + run) % 97}, weekly sales {i % 31}, lead days {i % 14 + 1}."
            for i in range(size)
        )
        yield (
            f"inventory-{size}",
            prefix + f"Using the inventory below, report the stock of product {first:04d} and the lead "
            f"days of product {second:04d}. Return only a JSON object with integer values "
            f"for 'stock' and 'lead_days'.\n\n{inventory}",
            {"stock": (first + run) % 97, "lead_days": second % 14 + 1},
        )


def meets_check(answer, expected, finish_reason="stop"):
    """Check specific fixture requirements, with the prose check limited to required terms.

    >>> meets_check('323', '323')
    True
    >>> meets_check('{"stock":4,"lead_days":7}', {"stock": 4, "lead_days": 7})
    True
    >>> meets_check('not JSON', {"stock": 4})
    False
    >>> meets_check('{"stock": 4.0}', {"stock": 4})
    False
    >>> meets_check('The lead time matters for the reorder threshold.', None, 'length')
    False
    """
    if finish_reason != "stop":
        return False
    if isinstance(expected, dict):
        answer = answer.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            values = json.loads(answer)
        except json.JSONDecodeError:
            return False
        return values == expected and all(type(values[key]) is int for key in expected)
    if expected is None:
        return all(term in answer.lower() for term in ("lead time", "reorder threshold"))
    return answer == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int)
    parser.add_argument("run", type=int, help="use the same run number for both engines")
    args = parser.parse_args()
    model = "inkling-2-mini"
    smoke, ready = generate("127.0.0.1", args.port, model, "Reply with exactly READY.", 512)
    if not meets_check(ready.strip(), "READY", smoke["finish_reason"]):
        raise RuntimeError(f"Model smoke check expected READY, got {ready!r}")
    failed = False
    for name, prompt, expected in cases(args.run):
        metrics, text = generate("127.0.0.1", args.port, model, prompt, 1024)
        answer = text.strip()
        passed = meets_check(answer, expected, metrics["finish_reason"])
        failed |= not passed
        print(json.dumps({
            "case": name, "run": args.run,
            "prompt_sha256": sha256(prompt.encode()).hexdigest(),
            "answer_sha256": sha256(answer.encode()).hexdigest(),
            "answer": answer,
            "check_passed": passed,
            **metrics,
        }), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
