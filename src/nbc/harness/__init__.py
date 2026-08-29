"""The run: entrypoint, measurement, timing, aggregation, verdicts.

Reads the corpus from `data/*.jsonl`, writes `results/scores.jsonl`, and reaches the report
through `results/results.json`. Each of those seams is a file, one-way, never an import.
"""
