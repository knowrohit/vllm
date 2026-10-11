# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Annotated, Generic, TypeAlias, TypeVar

from pydantic import BeforeValidator, Field, model_validator

from vllm import PoolingParams
from vllm.config import ModelConfig
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.exceptions import VLLMValidationError
from vllm.renderers import TokenizeParams
from vllm.tasks import PoolingTask
from vllm.utils import random_uuid

from ..base.protocol import (
    ChatRequestMixin,
    ClassifyRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    EncodingRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
    PoolingBasicRequestMixin,
    reject_removed_pooling_parameters,
)


class PoolingCacheReadRequestMixin(OpenAIBaseModel):
    skip_reading_prefix_cache: bool | None = Field(
        default=None,
        description=(
            "Whether to skip prefix-cache reads. None uses the task default. "
            "For token_embed and token_classify, False opts into partial outputs: "
            "cached prefix tokens are omitted. Use num_cached_tokens in each "
            "response item to locate the uncached suffix. This requires a causal "
            "model with prefix caching enabled."
        ),
    )

    @model_validator(mode="after")
    def check_prefix_cache_output_metadata(self):
        if (
            self.skip_reading_prefix_cache is False
            and getattr(self, "encoding_format", None) == "bytes_only"
        ):
            raise VLLMValidationError(
                "Prefix-cache reads require output metadata; use float, base64 "
                "or bytes encoding instead of bytes_only.",
                parameter="encoding_format",
            )
        return self


class PoolingCompletionRequest(
    PoolingCacheReadRequestMixin,
    PoolingBasicRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    ClassifyRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
):
    task: PoolingTask | None = None

    def to_pooling_params(self):
        return PoolingParams(
            task=self.task,
            use_activation=self.use_activation,
            dimensions=self.dimensions,
            skip_reading_prefix_cache=self.skip_reading_prefix_cache,
        )


class PoolingChatRequest(
    PoolingCacheReadRequestMixin,
    PoolingBasicRequestMixin,
    ChatRequestMixin,
    EmbedRequestMixin,
    ClassifyRequestMixin,
    FixedMaxLenTokenizeParamsMixin,
):
    task: PoolingTask | None = None

    def to_pooling_params(self):
        return PoolingParams(
            task=self.task,
            use_activation=self.use_activation,
            dimensions=self.dimensions,
            skip_reading_prefix_cache=self.skip_reading_prefix_cache,
        )


T = TypeVar("T")


class IOProcessorRequest(PoolingBasicRequestMixin, EncodingRequestMixin, Generic[T]):
    data: T
    task: PoolingTask = "plugin"

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return self._build_pooling_tok_params(
            model_config,
            add_special_tokens=not model_config.is_encoder_decoder,
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
        )

    def to_pooling_params(self):
        return PoolingParams(
            task=self.task,
        )


class IOProcessorResponse(OpenAIBaseModel, Generic[T]):
    request_id: str | None = None
    """
    The request_id associated with this response
    """
    created_at: int = Field(default_factory=lambda: int(time.time()))

    data: T
    """
    When using plugins IOProcessor plugins, the actual output is generated
    by the plugin itself. Hence, we use a generic type for the response data
    """


PoolingRequest: TypeAlias = Annotated[
    PoolingCompletionRequest | PoolingChatRequest | IOProcessorRequest,
    BeforeValidator(reject_removed_pooling_parameters),
]


class PoolingResponseData(OpenAIBaseModel):
    index: int
    object: str = "pooling"
    data: list[list[float]] | list[float] | str
    num_cached_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of prompt tokens reused from the prefix cache.",
    )


class PoolingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"pool-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[PoolingResponseData]
    usage: UsageInfo


class PoolingBytesResponse(OpenAIBaseModel):
    content: list[bytes]
    headers: dict[str, str] | None = None
    media_type: str = "application/octet-stream"
