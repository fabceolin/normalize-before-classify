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

`timed_read.py` is that argument one rung further out. SC1's claim — a stranger understands the
question, the table and the caveats in under five minutes — is a claim about the README, so its
checker belongs beside the other two; and it is a claim about a **human**, so the check deliberately
stops short of the thing being claimed. It measures the page's reading load from the file, holds the
protocol and the record of every timed read, and refuses a record that has been edited to fit a
result. The reading itself is a person's, and `not yet run` is a state it reports rather than one it
fails on. Like `size_budget.py` it reads no `results.json`; unlike either sibling, the property it is
about lives outside this repository entirely, which is why the module gates everything around the
reader and nothing about the reader.
"""
