# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

DYNAMO_KV_EVENT_BLOCK_SIZE_KEY = "dynamo_kv_event_block_size"


def get_configured_kv_event_block_size(vllm_config: VllmConfig) -> int:
    """Return the configured KV event block size, falling back to vLLM's cache block size."""
    additional_config = vllm_config.additional_config or {}
    return additional_config.get(
        DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
        vllm_config.cache_config.block_size,
    )


def select_main_attention_block_size(
    group_metadata: list[dict[str, Any]],
    fallback_block_size: int,
    configured_hash_block_size: int | None = None,
) -> int:
    """Select the block-hash granularity used by vLLM KV events.

    vLLM hashes requests at ``cache_config.hash_block_size`` when it is set.
    Otherwise, for hybrid models, it uses the GCD of every KV-cache group's
    block size.  Selecting one attention group's block size is incorrect for
    models such as DeepSeek-V4: its scheduler aligns at 256 tokens while its
    native and offload events carry 4-token request-block hashes.

    The Mamba exception mirrors ``resolve_kv_cache_block_sizes`` in vLLM: a
    divergent Mamba group disables finer hashing and uses the scheduler LCM.
    """
    if not group_metadata:
        return fallback_block_size

    block_sizes = [
        block_size
        for group in group_metadata
        if isinstance((block_size := group.get("block_size")), int) and block_size > 0
    ]
    if not block_sizes:
        return fallback_block_size

    if configured_hash_block_size is not None:
        return configured_hash_block_size

    if any(
        group.get("kind") == "mamba" and group.get("block_size") != fallback_block_size
        for group in group_metadata
    ):
        return math.lcm(*block_sizes)

    return math.gcd(*block_sizes)


async def configure_kv_event_block_size(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> int:
    """Fetch engine cache-group metadata and cache the KV event block size on vLLM config."""
    fallback_block_size = vllm_config.cache_config.block_size
    try:
        group_metadata = await engine.engine_core.call_utility_async(
            "get_kv_cache_group_metadata"
        )
    except Exception as e:
        logger.warning(
            "Failed to fetch KV cache group metadata; falling back to "
            "vLLM cache_config.block_size: %s",
            e,
        )
        kv_event_block_size = fallback_block_size
    else:
        kv_event_block_size = select_main_attention_block_size(
            group_metadata,
            fallback_block_size,
            getattr(vllm_config.cache_config, "hash_block_size", None),
        )

    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    return kv_event_block_size
