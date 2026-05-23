#!/usr/bin/env python3
"""Run the current sentence-level pipeline for Pet_Supplies."""

from __future__ import annotations

import os


os.environ["PQ_CATEGORY"] = "Pet_Supplies"

from train_vades_lite_sentence_latent_threshold import main  # noqa: E402


if __name__ == "__main__":
    main()
