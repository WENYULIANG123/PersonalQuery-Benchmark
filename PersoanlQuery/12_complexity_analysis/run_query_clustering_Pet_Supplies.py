#!/usr/bin/env python3
"""Run query-only PCA+GMM clustering for Pet_Supplies and write labels back."""

from __future__ import annotations

from cluster_strict5550_query_gmm_and_attach_retrieval import run_query_gmm_pipeline


if __name__ == "__main__":
    run_query_gmm_pipeline(
        category="Pet_Supplies",
        query_file=None,
        write_back_to_query_file=False,
        attach_retrieval=True,
    )
