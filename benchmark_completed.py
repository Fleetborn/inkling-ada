"""Compare completed Inkling answers over the loopback OpenAI API."""

import argparse
from hashlib import sha256
from http.client import HTTPException
import json
from time import perf_counter

from benchmark import generate


def inventory_case(run, size, prefix):
    first, second = size // 3, size - 4
    inventory = "\n".join(
        f"Product {i:04d}: stock {(i + run) % 97}, weekly sales {i % 31}, lead days {i % 14 + 1}."
        for i in range(size)
    )
    return (
        f"inventory-{size}",
        prefix + f"Using the inventory below, report the stock of product {first:04d} and the lead "
        f"days of product {second:04d}. Return only a JSON object with integer values "
        f"for 'stock' and 'lead_days'.\n\n{inventory}",
        {"stock": (first + run) % 97, "lead_days": second % 14 + 1},
    )


def cases(run):
    """Build paired prompts and expected answers for one run.

    >>> copy, edit = list(cases(2))[-2:]
    >>> copy[1].endswith(copy[2]) and meets_check(copy[2], copy[2])
    True
    >>> meets_check(copy[2].splitlines()[0], copy[2])
    False
    >>> original = edit[1].split("\\n\\n", 1)[1]
    >>> meets_check(original.replace("stock < threshold", "stock <= threshold"), edit[2])
    True
    >>> meets_check(original, edit[2]) or meets_check(edit[2], edit[2], "length")
    False
    """
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
        yield inventory_case(run, size, prefix)

    handoff = (
        f"Receiving ledger, shift {run}. The morning crew counted the sealed cartons twice "
        "and moved them to the blue rack beside the loading door.\n"
        "The packing list calls the small brass fitting a sleeve. On the shelf label it "
        "is called a collar. Both names refer to the same part.\n"
        "Leave the open carton on the inspection bench until the second delivery arrives. "
        "Record the two delivery numbers on separate lines before releasing the stock."
    )
    yield (
        "verbatim-copy",
        prefix + "Copy the complete handoff below verbatim. Preserve its line breaks and "
        "punctuation. Return only the handoff.\n\n" + handoff,
        handoff,
    )
    source = (
        "def reorder_list(products):\n"
        f"    default_threshold = {10 + run}\n"
        "    orders = []\n"
        "    for product in products:\n"
        '        stock = product["stock"]\n'
        '        threshold = product.get("threshold", default_threshold)\n'
        "        if stock < threshold:\n"
        '            quantity = product["target"] - stock\n'
        "            if quantity > 0:\n"
        '                orders.append((product["sku"], quantity))\n'
        "    return orders"
    )
    yield (
        "code-edit",
        prefix + "Edit this Python function so stock equal to the threshold also triggers "
        "a reorder. Change only the comparison operator. Return the entire function with its "
        "original formatting, without Markdown fences or commentary.\n\n" + source,
        source.replace("stock < threshold", "stock <= threshold"),
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


def warmup(port, model, timeout=120):
    metrics, ready = generate("127.0.0.1", port, model, "Reply with exactly READY.", 512, timeout)
    if not meets_check(ready.strip(), "READY", metrics["finish_reason"]):
        raise RuntimeError(f"Model smoke check expected READY, got {ready!r}")


def context_sweep(port, run, model):
    warmup_error = None
    try:
        warmup(port, model, timeout=600)
    except (OSError, HTTPException, RuntimeError, ValueError) as error:
        warmup_error = f"Warmup failed: {error}"
    passed_all = warmup_error is None
    for size in (105, 420, 780, 1580, 3200, 7500):
        name, prompt, expected = inventory_case(run, size, f"{size}-{run} inventory exercise. ")
        expected_tokens = 20 * size + 59  # Approximation from the historical inventory prompts.
        record = {
            "case": name, "run": run, "model": model, "rows": size,
            "prompt_sha256": sha256(prompt.encode()).hexdigest(),
            "expected_prompt_tokens": expected_tokens, "expected": expected, "check_passed": False,
        }
        if warmup_error is not None:
            print(json.dumps({**record, "error": warmup_error}), flush=True)
            continue
        start = perf_counter()
        try:
            metrics, text = generate("127.0.0.1", port, model, prompt, 1024, timeout=600)
            record.update(metrics)
            answer = text.strip()
            record.update(answer=answer, answer_sha256=sha256(answer.encode()).hexdigest())
            if abs(metrics["prompt_tokens"] - expected_tokens) > 32:
                record["error"] = "Reported prompt length outside expected range"
            else:
                record["check_passed"] = meets_check(answer, expected, metrics["finish_reason"])
        except (OSError, HTTPException, RuntimeError, ValueError) as error:
            record.update(error=str(error), elapsed_s=perf_counter() - start)
        print(json.dumps(record), flush=True)
        passed_all &= record["check_passed"]
    return passed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int)
    parser.add_argument("run", type=int, help="use the same run number for both engines")
    parser.add_argument("--model-id", default="inkling-2-mini")
    parser.add_argument("--context-sweep", action="store_true", help="test retrieval at six prompt lengths")
    args = parser.parse_args()
    model = args.model_id
    if args.context_sweep:
        if not context_sweep(args.port, args.run, model):
            raise SystemExit(1)
        return
    warmup(args.port, model)
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
