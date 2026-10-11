# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import requests
import torch

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.pooling.protocol import PoolingResponse

MODEL_NAME = "jason9693/Qwen2.5-1.5B-apeach"
DTYPE = "float32"  # Use float32 to avoid NaN issue
input_text = "This product was excellent and exceeded my expectations"
input_tokens = [1986, 1985, 572, 9073, 323, 33808, 847, 16665]


@pytest.fixture(scope="module")
def server():
    args = [
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--dtype",
        DTYPE,
        "--pooler-config.task",
        "token_classify",
        "--enable-prefix-caching",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def test_opt_in_prefix_cache_returns_aligned_classifications(server):
    body = {
        "model": MODEL_NAME,
        "input": "This product was excellent. " * 40 + "Read this tail.",
        "task": "token_classify",
    }

    def encode(**options):
        response = requests.post(server.url_for("pooling"), json={**body, **options})
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
        atol=1e-3,
        rtol=1e-3,
    )
    assert len(default["data"]) == len(full["data"])


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_pooling_token_classify(server: RemoteOpenAIServer, model_name: str):
    task = "token_classify"
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_text,
            "encoding_format": "float",
            "task": task,
        },
    )
    poolings = PoolingResponse.model_validate(response.json())
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == 8
    assert isinstance(poolings.data[0].data[0], list)
    assert len(poolings.data[0].data[0]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("task", ["classify", "embed", "token_embed", "plugin"])
async def test_pooling_not_supported(
    server: RemoteOpenAIServer, model_name: str, task: str
):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_text,
            "encoding_format": "float",
            "task": task,
        },
    )
    assert response.json()["error"]["type"] == "BadRequestError"

    if task == "plugin":
        err_msg = "No IOProcessor plugin installed."
    elif task == "classify":
        err_msg = "Try switching the model's pooling_task via"
    else:
        err_msg = f"Unsupported task: {task!r}"
    assert response.json()["error"]["message"].startswith(err_msg)
