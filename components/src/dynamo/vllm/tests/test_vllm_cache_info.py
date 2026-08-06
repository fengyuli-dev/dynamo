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


def test_hybrid_groups_use_request_hash_granularity() -> None:
    """DSv4 KV events use the GCD, not the 256-token scheduler alignment."""
    group_metadata = [
        {"kind": "mla_attention", "block_size": 256},
        {"kind": "sliding_window_mla", "block_size": 4},
        {"kind": "sliding_window_mla", "block_size": 8},
    ]

    assert select_main_attention_block_size(group_metadata, 256) == 4


def test_sink_only_metadata_uses_fallback() -> None:
    """A single valid group reports its own event granularity."""
    group_metadata = [
        {"kind": "sink_full_attention", "block_size": 256},
    ]

    assert select_main_attention_block_size(group_metadata, 16) == 256


def test_configured_hash_block_size_wins() -> None:
    """An explicit vLLM hash-block override is authoritative."""
    group_metadata = [
        {"kind": "full_attention", "block_size": 256},
        {"kind": "sliding_window", "block_size": 64},
    ]

    assert select_main_attention_block_size(group_metadata, 256, 32) == 32


def test_divergent_mamba_group_uses_scheduler_lcm() -> None:
    """Match vLLM's no-fine-hashing fallback for divergent Mamba groups."""
    group_metadata = [
        {"kind": "full_attention", "block_size": 16},
        {"kind": "mamba", "block_size": 1},
    ]

    assert select_main_attention_block_size(group_metadata, 16) == 16


def test_invalid_metadata_uses_fallback() -> None:
    """Missing or non-positive sizes cannot define a hash granularity."""
    group_metadata = [
        {"kind": "full_attention", "block_size": None},
        {"kind": "sliding_window", "block_size": 0},
    ]

    assert select_main_attention_block_size(group_metadata, 256) == 256
