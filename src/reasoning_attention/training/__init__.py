"""Stage-1 (SFT warm-start) training for the AV and AR.

The two halves train in opposite directions and therefore have different losses:

  - **AV** `h -> s`: causal LM cross-entropy on the summary tokens only, with the
    activation injected at the placeholder's embedding slot.
  - **AR** `s -> h`: MSE between the affine-mapped last-token residual and the
    target activation.

`data.py` holds the parquet datasets and collators; `sft.py` holds the loop.
"""

from reasoning_attention.training.data import ARBatch, ARDataset, AVBatch, AVDataset

__all__ = ["ARBatch", "ARDataset", "AVBatch", "AVDataset"]
