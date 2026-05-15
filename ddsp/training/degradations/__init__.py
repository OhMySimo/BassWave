# Copyright 2026 The DDSP Authors.
"""On-the-fly audio degradations for BassWave training augmentation.

Classes are imported here from their source files (which are kept clean of
gin decorators) and re-registered with `module=__name__` so that gin can
resolve them via short selectors like `@degradations.DegradationPipeline()`.
This mirrors the canonical DDSP pattern in `ddsp.training.models.__init__`.
"""

import gin

from ddsp.training.degradations.aggressive_compression import (
    AggressiveCompression)
from ddsp.training.degradations.bandwidth_limit import BandwidthLimit
from ddsp.training.degradations.base import BaseDegradation
from ddsp.training.degradations.degradation_main import DegradationPipeline
from ddsp.training.degradations.dynamic_eq import DynamicEQ
from ddsp.training.degradations.ghosting import Ghosting
from ddsp.training.degradations.phasing import Phasing
from ddsp.training.degradations.wrong_eq import WrongEQ


# Register every layer at the `ddsp.training.degradations` module level so
# gin selectors like `@degradations.X()` resolve correctly. Without this,
# each class would be registered under e.g.
# `ddsp.training.degradations.phasing.Phasing`, which gin matches against
# the suffix `phasing.Phasing` (not `degradations.Phasing`).
_configurable = lambda cls: gin.configurable(cls, module=__name__)

AggressiveCompression = _configurable(AggressiveCompression)
BandwidthLimit = _configurable(BandwidthLimit)
DegradationPipeline = _configurable(DegradationPipeline)
DynamicEQ = _configurable(DynamicEQ)
Ghosting = _configurable(Ghosting)
Phasing = _configurable(Phasing)
WrongEQ = _configurable(WrongEQ)
# BaseDegradation is abstract; not configurable.
