# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import requests
import torch

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.pooling.protocol import PoolingResponse

MODEL_NAME = "intfloat/multilingual-e5-small"
DTYPE = "bfloat16"
input_text = "The best thing about vLLM is that it supports many different models"
input_tokens = [
    0,
    581,
    2965,
    13580,
    1672,
    81,
    23708,
    594,
    83,
    450,
    442,
    8060,
    7,
    5941,
    12921,
    115774,
    2,
]


@pytest.fixture(scope="module")
def causal_server():
    args = [
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--pooler-config.task",
        "token_embed",
        "--max-model-len",
        "1024",
        "--enforce-eager",
        "--enable-prefix-caching",
    ]
    with RemoteOpenAIServer("Qwen/Qwen3-0.6B", args) as remote_server:
        yield remote_server


@pytest.mark.parametrize("chat", [False, True])
def test_opt_in_prefix_cache_returns_aligned_suffix(causal_server, chat):
    text = "A shared prefix for pooling requests. " * 40 + "Read this tail."
    body = {"model": "Qwen/Qwen3-0.6B", "task": "token_embed"}
    if chat:
        body["messages"] = [{"role": "user", "content": text}]
    else:
        body["input"] = text

    def encode(**options):
        response = requests.post(
            causal_server.url_for("pooling"), json={**body, **options}
        )
        assert response.status_code == 200, response.text
        return response.json()["data"][0]

    full = encode()
    partial = encode(skip_reading_prefix_cache=False)
    default = encode()
    offset = partial["num_cached_tokens"]
    assert 0 < offset < len(full["data"])
    assert full["num_cached_tokens"] == default["num_cached_tokens"] == 0
    torch.testing.assert_close(
        torch.tensor(partial["data"]),
        torch.tensor(full["data"])[offset:],
        atol=2e-2,
        rtol=2e-2,
    )
    assert len(default["data"]) == len(full["data"])


def test_partial_outputs_reject_bytes_only(causal_server):
    response = requests.post(
        causal_server.url_for("pooling"),
        json={
            "input": "hello",
            "task": "token_embed",
            "skip_reading_prefix_cache": False,
            "encoding_format": "bytes_only",
        },
    )
    assert response.status_code == 400, response.text
    assert "require output metadata" in response.json()["error"]["message"]


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--dtype",
        DTYPE,
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--pooler-config.task",
        "token_embed",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_pooling_token_embed(server: RemoteOpenAIServer, model_name: str):
    task = "token_embed"
    request_id = "client-request-id"
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_text,
            "encoding_format": "float",
            "task": task,
            "request_id": request_id,
        },
    )

    poolings = PoolingResponse.model_validate(response.json())

    assert poolings.id == f"pooling-{request_id}"
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == len(input_tokens)
    assert isinstance(poolings.data[0].data[0], list)
    assert len(poolings.data[0].data[0]) == 384


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("task", ["embed", "classify", "token_classify", "plugin"])
async def test_pooling_not_supported(
    server: RemoteOpenAIServer, model_name: str, task: str
):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": "test",
            "encoding_format": "float",
            "task": task,
        },
    )
    assert response.json()["error"]["type"] == "BadRequestError"

    if task == "plugin":
        err_msg = "No IOProcessor plugin installed."
    elif task == "embed":
        err_msg = "Try switching the model's pooling_task via"
    else:
        err_msg = f"Unsupported task: {task!r}"
    assert response.json()["error"]["message"].startswith(err_msg)
