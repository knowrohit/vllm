# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import importlib.util
import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.entrypoints.pooling.utils import (
    MetadataItem,
    decode_pooling_output,
    encode_pooling_bytes,
    encode_pooling_output_float_or_ndarray,
)
from vllm.outputs import PoolingOutput, PoolingRequestOutput


def _pooling_output(data):
    return SimpleNamespace(outputs=SimpleNamespace(data=data))


@pytest.mark.parametrize("encoding", ["float", "base64", "bytes"])
def test_partial_pooling_response_preserves_per_item_cache_offsets(encoding):
    from fastapi.responses import JSONResponse

    from vllm.entrypoints.pooling.pooling.serving import ServingPooling

    outputs = [
        PoolingRequestOutput(
            request_id=str(i),
            outputs=PoolingOutput(torch.arange(6, dtype=torch.float32).reshape(3, 2)),
            prompt_token_ids=[1] * (3 + cached),
            num_cached_tokens=cached,
            finished=True,
        )
        for i, cached in enumerate([0, 16])
    ]
    if encoding == "bytes":
        bodies, items = encode_pooling_bytes(outputs, "float32", "little")
        assert [item["num_cached_tokens"] for item in items] == [0, 16]
        decoded = decode_pooling_output(
            [MetadataItem(**item) for item in items], b"".join(bodies)
        )
        for actual, original in zip(decoded, outputs):
            torch.testing.assert_close(actual, original.outputs.data)
    else:
        serving = ServingPooling.__new__(ServingPooling)
        serving.json_response_cls = JSONResponse
        response = serving.request_output_to_pooling_json_response(
            outputs, "pool-test", 0, "test-model", encoding, "float32", "little"
        )
        payload = json.loads(response.body)
        assert [item["num_cached_tokens"] for item in payload["data"]] == [0, 16]
        assert payload["usage"]["prompt_tokens"] == 22


def test_encode_pooling_output_float_or_ndarray_returns_numpy_array():
    output = _pooling_output(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    encoded = encode_pooling_output_float_or_ndarray(output)

    assert isinstance(encoded, np.ndarray)
    np.testing.assert_allclose(encoded, [1.0, 2.0, 3.0])


@pytest.mark.skipif(
    importlib.util.find_spec("orjson") is None,
    reason="orjson is not installed",
)
def test_orjson_serializes_numpy_array():
    from fastapi.responses import ORJSONResponse

    output = _pooling_output(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    encoded = encode_pooling_output_float_or_ndarray(output)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        response = ORJSONResponse(content={"embedding": encoded})
    assert json.loads(response.body)["embedding"] == pytest.approx([1.0, 2.0, 3.0])


def test_encode_pooling_output_float_or_ndarray_falls_back_to_list():
    class DataWithUnsupportedNumpy:
        def is_contiguous(self):
            return True

        def numpy(self):
            raise TypeError("unsupported dtype")

        def tolist(self):
            return [1.0, 2.0, 3.0]

    output = _pooling_output(DataWithUnsupportedNumpy())

    assert encode_pooling_output_float_or_ndarray(output) == [1.0, 2.0, 3.0]
