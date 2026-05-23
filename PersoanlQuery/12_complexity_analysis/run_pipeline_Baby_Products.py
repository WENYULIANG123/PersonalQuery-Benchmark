#!/usr/bin/env python3
"""Run the current sentence-level pipeline for Baby_Products."""

from __future__ import annotations

import os


os.environ["PQ_CATEGORY"] = "Baby_Products"

from train_vades_lite_sentence_latent_threshold import main  # noqa: E402


if __name__ == "__main__":
    main()
