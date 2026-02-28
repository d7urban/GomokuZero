#!/usr/bin/env python3
"""Migrate GomokuZero weights from 6 residual blocks to 10 residual blocks.

The migration keeps all learned tensors from the 6-block, 128-filter model,
including all 6 input feature planes. The extra 4 residual blocks are
initialized as near-identity blocks so the migrated network starts from the
same behaviour and can continue training with deeper capacity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gomoku import NUM_INPUT_PLANES, create_model


_STEM_TENSOR_COUNT = 5
_POLICY_HEAD_TENSOR_COUNT = 7
_VALUE_HEAD_TENSOR_COUNT = 9


def _block_tensor_count(block_idx: int) -> int:
    return 14 if (block_idx % 2 == 1) else 10


def _split_weights(weights: list[np.ndarray], num_blocks: int) -> dict[str, list[list[np.ndarray]] | list[np.ndarray]]:
    idx = 0
    stem = [w.copy() for w in weights[idx:idx + _STEM_TENSOR_COUNT]]
    idx += _STEM_TENSOR_COUNT

    blocks: list[list[np.ndarray]] = []
    for block_idx in range(num_blocks):
        count = _block_tensor_count(block_idx)
        blocks.append([w.copy() for w in weights[idx:idx + count]])
        idx += count

    policy = [w.copy() for w in weights[idx:idx + _POLICY_HEAD_TENSOR_COUNT]]
    idx += _POLICY_HEAD_TENSOR_COUNT

    value = [w.copy() for w in weights[idx:idx + _VALUE_HEAD_TENSOR_COUNT]]
    idx += _VALUE_HEAD_TENSOR_COUNT

    if idx != len(weights):
        raise ValueError(
            f"Unexpected tensor count: consumed {idx}, but model has {len(weights)} tensors."
        )

    return {
        "stem": stem,
        "blocks": blocks,
        "policy": policy,
        "value": value,
    }


def _neutralize_block(block: list[np.ndarray], has_se: bool) -> list[np.ndarray]:
    out = [w.copy() for w in block]

    # Keep conv1/bn1 as initialized to preserve gradient flow.
    # Set conv2 kernel to zero so the residual branch starts as exact identity:
    #   y = ReLU(residual + 0)
    # Crucially, conv2 remains trainable on the first update step.
    out[5] = np.zeros_like(out[5])  # conv2 kernel
    out[6] = np.ones_like(out[6])   # bn2 gamma
    out[7] = np.zeros_like(out[7])  # bn2 beta
    out[8] = np.zeros_like(out[8])  # bn2 moving mean
    out[9] = np.ones_like(out[9])   # bn2 moving var

    # For SE blocks we keep Dense layers as initialized; with conv2=0 they are
    # functionally neutral at start, and they can learn once conv2 activates.
    _ = has_se

    return out


def migrate_weights(source: Path, target: Path) -> None:
    src_model = create_model(num_res_blocks=6, num_filters=128)
    src_model.load_weights(str(source))

    src_weights = src_model.get_weights()
    src_parts = _split_weights(src_weights, num_blocks=6)

    stem_kernel = src_parts["stem"][0]
    if stem_kernel.shape[2] != NUM_INPUT_PLANES:
        raise ValueError(
            "Source weights do not match the current 6-plane encoder. "
            f"Found {stem_kernel.shape[2]} input planes, expected {NUM_INPUT_PLANES}."
        )

    dst_model = create_model(num_res_blocks=10, num_filters=128)
    dst_parts = _split_weights(dst_model.get_weights(), num_blocks=10)

    dst_parts["stem"] = [w.copy() for w in src_parts["stem"]]
    for block_idx in range(10):
        if block_idx < 6:
            dst_parts["blocks"][block_idx] = [w.copy() for w in src_parts["blocks"][block_idx]]
        else:
            has_se = (block_idx % 2 == 1)
            dst_parts["blocks"][block_idx] = _neutralize_block(
                dst_parts["blocks"][block_idx], has_se=has_se
            )

    dst_parts["policy"] = [w.copy() for w in src_parts["policy"]]
    dst_parts["value"] = [w.copy() for w in src_parts["value"]]

    final_weights: list[np.ndarray] = []
    final_weights.extend(dst_parts["stem"])
    for block in dst_parts["blocks"]:
        final_weights.extend(block)
    final_weights.extend(dst_parts["policy"])
    final_weights.extend(dst_parts["value"])

    dst_model.set_weights(final_weights)

    target.parent.mkdir(parents=True, exist_ok=True)
    dst_model.save_weights(str(target))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate 6-block (128-filter) GomokuZero weights to 10-block (128-filter)."
    )
    parser.add_argument("source", type=Path, help="Path to source .weights.h5 file (6-block model)")
    parser.add_argument("target", type=Path, help="Path for migrated .weights.h5 file (10-block model)")
    args = parser.parse_args()

    migrate_weights(args.source, args.target)
    print(f"Migrated weights saved to: {args.target}")


if __name__ == "__main__":
    main()
