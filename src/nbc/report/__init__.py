"""The sink: renders the table and the verdicts from `results.json`, and reads nothing else.

A number in the README that is not in `results.json` is a number no one can trace.

The README's own structure belongs here too, and is not a counter-example: `caveats.py` verifies
the hand-written honesty section, and `readme.py` will inject the generated table between the
`RESULTS` markers. Neither carries a number — one reads prose the run must never write, the other
writes the block a human must never edit.
"""
