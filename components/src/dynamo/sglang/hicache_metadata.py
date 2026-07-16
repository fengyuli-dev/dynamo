# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional

_DSV4_C4 = "deepseek_v4_c4"
_DSV4_C4_INDEXER = "deepseek_v4_c4_indexer"
_DSV4_C128 = "deepseek_v4_c128"
_DSV4_C4_STATE = "deepseek_v4_c4_state"
_DSV4_C4_INDEXER_STATE = "deepseek_v4_c4_indexer_state"


def get_deepseek_v4_mooncake_page_layout(
    engine: Any, server_args: Any
) -> Optional[dict[str, Any]]:
    """Describe the physical Mooncake objects behind one DSV4 logical page.

    DeepSeek V4's full-KV host pool is a logical allocation anchor. Its
    physical data is stored in compressed-KV sidecars plus a trailing SWA
    window. Publishing that layout lets the router query objects that actually
    exist instead of the generic MLA ``{page_hash}__k`` key.
    """
    try:
        hf_config = engine.tokenizer_manager.model_config.hf_config
        model_type = str(getattr(hf_config, "model_type", "")).lower()
        architectures = {
            str(architecture).lower()
            for architecture in (getattr(hf_config, "architectures", None) or [])
        }
    except Exception as e:
        logging.warning(f"Failed to inspect model config for Mooncake layout: {e}")
        return None

    if model_type != "deepseek_v4" and not any(
        "deepseekv4" in architecture for architecture in architectures
    ):
        return None

    try:
        compression_ratios = {
            int(ratio) for ratio in (getattr(hf_config, "compress_ratios", None) or [])
        }
    except (TypeError, ValueError) as e:
        logging.warning(f"Invalid DeepSeek V4 compress_ratios; ignoring layout: {e}")
        return None

    # These values mirror SGLang's PoolName wire values. Keeping this helper
    # import-light is important because runtime-metadata tests run without CUDA.
    all_page_pools: list[str] = []
    trailing_page_pools = ["swa"]
    if 4 in compression_ratios:
        all_page_pools.extend([_DSV4_C4, _DSV4_C4_INDEXER])
        trailing_page_pools.extend([_DSV4_C4_STATE, _DSV4_C4_INDEXER_STATE])
    if 128 in compression_ratios:
        all_page_pools.append(_DSV4_C128)

    if not all_page_pools:
        logging.warning(
            "DeepSeek V4 Mooncake layout has no compressed-KV pools; "
            "falling back to generic MLA lookup."
        )
        return None

    try:
        page_size = max(1, int(getattr(server_args, "page_size", 1) or 1))
        pp_size = max(1, int(getattr(server_args, "pp_size", 1) or 1))
        sliding_window = max(
            1, int(getattr(hf_config, "sliding_window", page_size) or page_size)
        )
    except (TypeError, ValueError) as e:
        logging.warning(f"Invalid DeepSeek V4 page/window layout; ignoring it: {e}")
        return None

    trailing_page_count = max(1, (sliding_window + page_size - 1) // page_size)

    def pool_suffixes(pool_name: str) -> list[str]:
        if pp_size > 1:
            return [f"_{pp_rank}_{pool_name}" for pp_rank in range(pp_size)]
        return [f"__{pool_name}"]

    return {
        "all_pages_suffixes": [
            suffix for pool in all_page_pools for suffix in pool_suffixes(pool)
        ],
        "trailing_pages_suffixes": [
            suffix for pool in trailing_page_pools for suffix in pool_suffixes(pool)
        ],
        "trailing_page_count": trailing_page_count,
    }
