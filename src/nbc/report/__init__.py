"""The sink: renders the table and the verdicts from `results.json`, and reads nothing else.

A number in the README that is not in `results.json` is a number no one can trace.

The README's own structure belongs here too, and is not a counter-example: `caveats.py` verifies
the hand-written honesty section, and `readme.py` will inject the generated table between the
`RESULTS` markers. Neither carries a number — one reads prose the run must never write, the other
writes the block a human must never edit.

`size_budget.py` is the one module here that reads neither `results.json` nor the run: it measures
the canonicalization layer as text and compares the result to the budget the README states and the
constant `SizeBudget` declares. It lives here because it is a check on a README claim, which is
what `caveats.py` is, and it is deliberately **not** in `canon/` — a module inside the thing it
measures would be a module that grows the number it is bounding.
"""
