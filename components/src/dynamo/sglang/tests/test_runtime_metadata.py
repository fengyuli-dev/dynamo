# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.sglang.capacity import get_spec_decode_runtime_data
from dynamo.sglang.hicache_metadata import get_deepseek_v4_mooncake_page_layout

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_spec_decode_runtime_data_uses_speculative_num_steps():
    server_args = SimpleNamespace(
        speculative_num_steps="5",
        speculative_algorithm="EAGLE",
    )

    assert get_spec_decode_runtime_data(server_args) == {
        "nextn": 5,
        "method": "EAGLE",
        "source": "backend_config",
    }


@pytest.mark.parametrize(
    "speculative_num_steps",
    [None, 0, "bad"],
)
def test_spec_decode_runtime_data_ignores_invalid_nextn(speculative_num_steps):
    server_args = SimpleNamespace(
        speculative_num_steps=speculative_num_steps,
        speculative_algorithm="EAGLE",
    )

    assert get_spec_decode_runtime_data(server_args) is None


def test_deepseek_v4_mooncake_page_layout_matches_physical_pools():
    hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        compress_ratios=[128, 4, 128, 4, 0],
        sliding_window=128,
    )
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf_config)
        )
    )
    server_args = SimpleNamespace(page_size=256, pp_size=1)

    assert get_deepseek_v4_mooncake_page_layout(engine, server_args) == {
        "all_pages_suffixes": [
            "__deepseek_v4_c4",
            "__deepseek_v4_c4_indexer",
            "__deepseek_v4_c128",
        ],
        "trailing_pages_suffixes": [
            "__swa",
            "__deepseek_v4_c4_state",
            "__deepseek_v4_c4_indexer_state",
        ],
        "trailing_page_count": 1,
    }


def test_non_deepseek_v4_has_no_hybrid_mooncake_layout():
    hf_config = SimpleNamespace(
        model_type="deepseek_v3",
        architectures=["DeepseekV3ForCausalLM"],
    )
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf_config)
        )
    )

    assert (
        get_deepseek_v4_mooncake_page_layout(
            engine, SimpleNamespace(page_size=256, pp_size=1)
        )
        is None
    )
