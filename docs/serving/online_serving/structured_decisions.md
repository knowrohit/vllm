# Structured Decisions

The `/v1/systemone` endpoint answers a set of typed questions about a state and
returns a probability for every allowed answer.

The endpoint is available by default on generation models. Models without a
supported read strategy return 501. Supported architectures require
`--logprobs-mode raw_logprobs` or `processed_logprobs`, since `label_mass` needs
log probabilities.

## How it works

1. Each question is one read. The user message is the state, followed by the
   question and its options with the type's labels, and an instruction to
   answer with one label.
2. The read is the reply's first token. Each label must be one token there,
   or the request returns 400.
3. The read returns the logprobs of that question's label tokens. Softmax over
   the labels gives the answer's probabilities.

Thinking is off unless `chat_template_kwargs` turns it on.

Probabilities can differ between identical requests. They depend on which
requests share a batch and on whether a read hits the prefix cache. Set
`VLLM_BATCH_INVARIANT=1` for reads that repeat exactly, at a cost in
throughput.

Every read starts with the state. With `--enable-prefix-caching`, reads can
reuse completed, token-identical prefixes and prefill only their uncached tails.
Simultaneous cold reads may each prefill the state before any cache entry is
available. Reuse also depends on the model's cache-hit alignment; sharing some
text does not guarantee a hit.

`diagnostics.<question>.cached_tokens` reports how many prompt tokens that read
reused. `usage.cached_tokens` sums these counts across reads, including prefixes
reused by multiple questions. For hybrid models, inspect the
`vllm:cache_config_info` labels on `/metrics`, especially `hash_block_size` and
`cache_hit_alignment_tokens`, rather than assuming a 16-token matching unit
from the physical block size. A shared token prefix shorter than the resolved
alignment cannot produce a hit.

## Question types

| type | criteria | answer |
| --- | --- | --- |
| `choice` | map of option name to a description or `null` | `choice`, `probabilities` by option name, `confidence` |
| `noul` | `null`, or an object with optional `true` and `false` descriptions | `noul` (probability of yes), `probabilities`, `confidence` |
| `score` | ordered list of level names | `score` (expected 0-indexed level), `legend`, `probabilities` by level index, `confidence` |

A choice has at least one option, a score at least two levels. A question id
is a non-empty string.

## Example

```bash
curl -s http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "state": {"ticket": "My card was charged twice for one order."},
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
          "billing": "payments and refunds",
          "shipping": "deliveries",
          "security": "account access"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does the customer need a reply within the hour?",
        "criteria": {"true": "the account is locked or money is moving"}
      },
      "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"]
      }
    },
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Response, with probabilities that depend on the model:

```json
{
  "id": "decision-...",
  "object": "structured_decision",
  "created": 1790000000,
  "model": "Qwen/Qwen3-0.6B",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.97, "shipping": 0.01, "security": 0.02},
      "confidence": 0.90
    },
    "urgent": {
      "type": "noul",
      "noul": 0.02,
      "probabilities": {"yes": 0.02, "no": 0.98},
      "confidence": 0.95
    },
    "tone": {
      "type": "score",
      "score": 0.6,
      "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
      "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1},
      "confidence": 0.65
    }
  },
  "usage": {"input_tokens": 142, "output_tokens": 3, "cached_tokens": 0},
  "diagnostics": {
    "team": {"label_mass": 0.93, "argmax_is_label": true, "cached_tokens": 0},
    "urgent": {"label_mass": 0.98, "argmax_is_label": true, "cached_tokens": 0},
    "tone": {"label_mass": 0.91, "argmax_is_label": true, "cached_tokens": 0}
  }
}
```

`diagnostics.label_mass` is the probability the model put on the labels across
its whole vocabulary. A low value means most of the probability went to tokens
that are not labels. `confidence` is the chosen option's probability over the
whole vocabulary: its share of the labels times `label_mass`.

## Labels

Labels follow the type: a `choice` labels its options `A` to `Z` in the order
the request lists them, a `noul` labels its options `yes` and `no`, and a
`score` labels its levels `0` to `9`, so a level's label is its score.

## Limits

| limit | value |
| --- | --- |
| questions per request | 64 |
| options per `choice` | 26 |
| levels per `score` | 10 |

## Request fields

| field | meaning |
| --- | --- |
| `state` | what the questions are about: a string, or JSON that is sent as its JSON text |
| `questions` | question id to `{type, instructions, criteria}`, asked in this order |
| `instructions` | optional context placed ahead of the questions |
| `chat_template_kwargs` | passed to the chat template, for example `{"enable_thinking": false}` |
| `cache_salt` | optional [prefix-cache salt](../../usage/security.md#cache-salting) applied to every question |

A question with any field other than `type`, `instructions` and `criteria` is
rejected with a 400.

When [tracing](../../../examples/observability/opentelemetry/README.md) is enabled
with `--otlp-traces-endpoint`, the HTTP `traceparent` and `tracestate` headers are
forwarded to every question's generation request.
