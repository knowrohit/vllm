# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check mismatch rejection through EngineArgs using a local tiny model config."""

import tempfile

from transformers import Qwen2Config

from vllm.engine.arg_utils import EngineArgs

with tempfile.TemporaryDirectory() as model_path:
    Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=256,
    ).save_pretrained(model_path)
    for model_dtype, cache_dtype in [
        ("bfloat16", "float16"),
        ("float16", "bfloat16"),
    ]:
        try:
            EngineArgs(
                model=model_path,
                dtype=model_dtype,
                kv_cache_dtype=cache_dtype,
                max_model_len=128,
            ).create_engine_config()
        except ValueError as error:
            expected = (
                f'kv_cache_dtype="{cache_dtype}" is incompatible '
                f'with dtype="{model_dtype}"'
            )
            assert expected in str(error), str(error)
            print(f"PASS: EngineArgs rejected {model_dtype} model + {cache_dtype} KV")
        else:
            raise AssertionError("EngineArgs accepted the mismatch")
