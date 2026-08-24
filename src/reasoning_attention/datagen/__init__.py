"""Warm-start (Stage 1 / SFT) dataset generation for the NLA.

Four stages, mirroring the reference repo's pipeline but specialized to a single
GPU and an OpenAI-compatible explainer:

    extract -> split -> explain -> build

`extract` forwards the target model over Ultra-FineWeb and stores RAW layer-l
activations. `split` partitions **by document** into the disjoint AV and AR
halves. `explain` calls the API model to write each row's summary. `build`
renders the final AV/AR training parquets.
"""

from reasoning_attention.datagen.providers import CompletionProvider, OpenAIProvider

__all__ = ["CompletionProvider", "OpenAIProvider"]
