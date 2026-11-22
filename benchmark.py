"""Compare single-stream Inkling 2 Mini inference over a loopback OpenAI API."""

import argparse
from hashlib import sha256
from http.client import HTTPConnection
import json
from time import perf_counter
from urllib.parse import urlsplit


def read_stream(lines, start):
    """Read assistant deltas and final usage.

    >>> lines = [b'data: {"choices":[{"delta":{"content":"READY"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1}}', b'data: [DONE]']
    >>> result, text = read_stream(lines, perf_counter())
    >>> (result["prompt_tokens"], result["completion_tokens"], text)
    (10, 1, 'READY')
    >>> reasoning = [lines[0].replace(b'"content":"READY"', b'"reasoning_content":"Thinking"'), lines[1]]
    >>> result, text = read_stream(reasoning, perf_counter())
    >>> (result["ttft_s"] >= 0, result["ttfa_s"], text)
    (True, None, '')
    >>> result, _ = read_stream([reasoning[0], *lines], perf_counter())
    >>> 0 <= result["ttft_s"] <= result["ttfa_s"] <= result["elapsed_s"]
    True
    >>> read_stream(lines[:1], perf_counter())
    Traceback (most recent call last):
    ...
    RuntimeError: Incomplete SSE stream
    """
    usage = timings = finish = first = first_answer = last = None
    output = []
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        event = json.loads(payload)
        if "error" in event:
            raise RuntimeError(f"Inference failed: {event['error']}")
        usage = event.get("usage") or usage
        timings = event.get("timings") or timings
        for choice in event.get("choices", []):
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content or delta.get("reasoning_content"):
                last = perf_counter()
                if first is None:
                    first = last
            if content:
                if first_answer is None:
                    first_answer = last
                output.append(content)
    else:
        raise RuntimeError("Incomplete SSE stream")

    end = perf_counter()
    if first is None or finish is None:
        raise RuntimeError("Missing streamed output or finish reason")
    if not usage or any(
        type(usage.get(key)) is not int or usage[key] <= 0
        for key in ("prompt_tokens", "completion_tokens")
    ):
        raise RuntimeError("Missing positive prompt or completion usage")
    # shortcut: SSE chunks may bundle tokens, use native timings for exact decode rates.
    decode = (usage["completion_tokens"] - 1) / (last - first) if last > first else None
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "finish_reason": finish,
        "ttft_s": first - start,
        "ttfa_s": first_answer - start if first_answer is not None else None,
        "elapsed_s": end - start,
        "decode_tok_s_est": decode,
        "server_timings": timings,
    }, "".join(output)


def generate(host, port, model, prompt, limit):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "none",
        "temperature": 0,
        "max_tokens": limit,
    }).encode()
    connection = HTTPConnection(host, port=port, timeout=120)
    start = perf_counter()
    try:
        connection.request("POST", "/v1/chat/completions", body, {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        })
        with connection.getresponse() as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {response.read(400)!r}")
            return read_stream(response, start)
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="e.g. http://127.0.0.1:8080")
    parser.add_argument("model")
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "::1")
        or url.path.rstrip("/") not in ("", "/v1")
        or url.username is not None
        or url.query
        or url.fragment
    ):
        parser.error("expected a loopback HTTP URL with an optional /v1 suffix")

    _, ready = generate(url.hostname, url.port, args.model, "Reply with exactly READY.", 512)
    if ready.strip() != "READY":
        raise RuntimeError(f"Model smoke check expected READY, got {ready!r}")

    for rows in (105, 420):
        inventory = "\n".join(
            f"Product {i:04d}: stock {i % 97}, weekly sales {i % 31}, lead days {i % 14 + 1}."
            for i in range(rows)
        )
        for run in range(1, 4):
            prompt = (
                f"{rows * run}-{run} inventory exercise. Write a 400-word training guide "
                "for a weekly reordering review. Use examples from this synthetic inventory.\n\n"
                f"{inventory}"
            )
            result, _ = generate(url.hostname, url.port, args.model, prompt, 128)
            print(json.dumps({
                "rows": rows, "run": run,
                "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                **result,
            }), flush=True)


if __name__ == "__main__":
    main()
