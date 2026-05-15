# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Denoising preprocessor: corrupts audio for the encoder pathway, preserves
the clean audio under `audio_clean` for the loss target.

The on-the-fly corruption is delegated to
`ddsp.training.degradations.DegradationPipeline`, which composes 6
configurable bass-specific degradations (phasing, ghosting, wrong EQ,
aggressive compression, dynamic EQ, random bandwidth limit) and randomly
selects K∈[min_n_apply, max_n_apply] of them per training step.
"""

import gin
import tensorflow.compat.v2 as tf

from ddsp.training import preprocessing
from ddsp.training.degradations.degradation_main import DegradationPipeline


@gin.register
class DenoisingF0LoudnessPreprocessor(preprocessing.F0LoudnessPreprocessor):
  """F0LoudnessPreprocessor + DegradationPipeline applied on-the-fly.

  Reads `audio` (clean from dataset). Stashes a copy under `audio_clean`,
  then overwrites `audio` with the corrupted version. Encoder reads
  `audio` (now corrupted), the spectral loss compares synth output against
  `audio_clean`.

  f0_hz and loudness_db come from the precomputed (clean) dataset values.
  At inference time on real degraded audio you'll need to provide f0
  externally (or run CREPE on the degraded input) — see README.
  """

  def __init__(self,
               degradation_pipeline=None,
               compute_loudness: bool = False,
               **kwargs):
    super().__init__(compute_loudness=compute_loudness, **kwargs)
    self.degradation_pipeline = degradation_pipeline or DegradationPipeline()

  def call(self, loudness_db, f0_hz, audio=None) -> [
      'f0_hz', 'loudness_db', 'f0_scaled', 'ld_scaled',
      'audio', 'audio_clean']:
    audio_clean = audio
    audio_corrupted = self.degradation_pipeline(audio, training=True)

    f0_hz = self.resample(f0_hz)
    loudness_db = self.resample(loudness_db)
    f0_scaled = preprocessing.scale_f0_hz(f0_hz)
    ld_scaled = preprocessing.scale_db(loudness_db)

    return (f0_hz, loudness_db, f0_scaled, ld_scaled,
            audio_corrupted, audio_clean)
