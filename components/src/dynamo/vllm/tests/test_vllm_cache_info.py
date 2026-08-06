#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM KV-event cache-group selection."""

import pytest

from dynamo.vllm.cache_info import select_main_attention_block_size

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_mla_group_wins_over_sink_full_group() -> None:
    """Draft/sink metadata must not override the main MLA hash granularity."""
    group_metadata = [
        {"kind": "sink_full_attention", "block_size": 256},
        {"kind": "mla_attention", "block_size": 4},
    ]

    assert select_main_attention_block_size(group_metadata, 256) == 4


def test_sink_only_metadata_uses_fallback() -> None:
    """Sink attention is not the router's main prefix-cache group."""
    group_metadata = [
        {"kind": "sink_full_attention", "block_size": 256},
    ]

    assert select_main_attention_block_size(group_metadata, 16) == 16
