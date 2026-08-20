#!/usr/bin/env bash
set -euo pipefail

workspace=/home/scratch.fengyul_coreai
dynamo_src=${workspace}/.tmp/dynamo-dsv4-vllm-nightly-build
vllm_src=${workspace}/.tmp/vllm-dsv4-nightly-minimal
base_image=deepseek-vllm-nightly-dynamo-base:57df13d
candidate_image=deepseek-vllm-nightly-dynamo:57df13d-d626108b-retention
dynamo_sha=57df13d89667f7133e712ba11725cd5a70652962
vllm_sha=d626108b1841888ec90aced33367149a6bbc7e4b
patch_head=895f6076473156a1f1e33e799112bc5f77f05242

test "$(git -C "${dynamo_src}" rev-parse "${dynamo_sha}^{commit}")" = "${dynamo_sha}"
test "$(git -C "${dynamo_src}" merge-base HEAD "${dynamo_sha}")" = "${dynamo_sha}"
expected_dynamo_changes=$'container/Dockerfile.dsv4-vllm-overlay\ncontainer/build-dsv4-vllm-nightly.sh\ncontainer/compliance/license_overrides.yaml\ncontainer/context.yaml'
test "$(git -C "${dynamo_src}" diff --name-only "${dynamo_sha}..HEAD")" = "${expected_dynamo_changes}"
test -z "$(git -C "${dynamo_src}" status --porcelain)"
dynamo_build_sha=$(git -C "${dynamo_src}" rev-parse HEAD)
test "$(git -C "${vllm_src}" rev-parse "${vllm_sha}^{commit}")" = "${vllm_sha}"
test "$(git -C "${vllm_src}" rev-parse HEAD)" = "${patch_head}"
test -z "$(git -C "${vllm_src}" status --porcelain)"

"${dynamo_src}/container/render.py" \
  --framework vllm \
  --target runtime \
  --output-short-filename
docker buildx build \
  --load \
  --progress=plain \
  --build-arg "DYNAMO_COMMIT_SHA=${dynamo_sha}" \
  -f "${dynamo_src}/container/rendered.Dockerfile" \
  -t "${base_image}" \
  "${dynamo_src}"

docker buildx build \
  --load \
  --progress=plain \
  --build-context "vllm_src=${vllm_src}" \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "DYNAMO_SOURCE_SHA=${dynamo_sha}" \
  --build-arg "DYNAMO_BUILD_SHA=${dynamo_build_sha}" \
  --build-arg "VLLM_SOURCE_SHA=${vllm_sha}" \
  --build-arg "VLLM_PATCH_HEAD=${patch_head}" \
  -f "${dynamo_src}/container/Dockerfile.dsv4-vllm-overlay" \
  -t "${candidate_image}" \
  "${dynamo_src}"

docker run --rm --entrypoint /bin/bash "${candidate_image}" -lc '
set -euo pipefail
python3 - <<"PY"
import importlib.metadata as metadata
import os

from vllm.config.speculative import SpeculativeConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.kv_offload.cpu.gpu_worker import MAX_HOST_REGISTER_CHUNK_BYTES
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    shift_draft_block_tables,
)

assert metadata.version("vllm")
try:
    metadata.version("vllm-omni")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("unused incompatible vllm-omni package is still installed")
assert os.environ["DYNAMO_COMMIT_SHA"] == (
    "57df13d89667f7133e712ba11725cd5a70652962"
)
assert hasattr(SpeculativeConfig, "has_ephemeral_draft_context")
assert "eagle_group_is_veto_exempt" in KVCacheGroupSpec.__dataclass_fields__
assert MAX_HOST_REGISTER_CHUNK_BYTES == 64 * 1024**3
assert callable(shift_draft_block_tables)
source = __import__("inspect").getsource(OffloadingConnectorScheduler)
assert "reachable_block_mask" in source
assert "first_fresh_gpu_block" in source
print("SOURCE_GATE_OK", metadata.version("vllm"))
PY
'

docker image inspect "${candidate_image}" \
  --format 'RESULT=BUILT image={{index .RepoTags 0}} id={{.Id}}'
