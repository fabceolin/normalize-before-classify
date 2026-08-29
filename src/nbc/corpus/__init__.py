"""The corpus builder: the only writer of `data/*.jsonl`.

Reaches the harness through those files and never by import, so one code path renders corpus
text and the gold labels cannot drift from the text they label.
"""
