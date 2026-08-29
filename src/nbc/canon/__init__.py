"""The canonicalization layer: ordered, pure text-to-text filters.

Imports only the standard library, its vendored data, and the leaf modules of `nbc` that import
nothing but the standard library themselves — today `nbc.schema` and `nbc.errors`. It must stay
usable in front of any classifier, with no model dependency of its own.

`nbc.errors` is in that allowance for a reason and not by drift. The repository rule is that every
abort raises from `nbc/errors.py` with an exit code no other abort shares, and a vendored
confusables table that disagrees with the interpreter's own Unicode revision is exactly such an
abort: it changes the meaning of every number the run publishes. `nbc.errors` is a leaf that
imports only `types` and `typing`, so admitting it costs the layer nothing that the bound exists to
protect — no model, no third-party package, no reach back into the harness.

The bound is enforced, not asserted: `tests/canon/test_import_bound.py` parses every module under
this package and every module in the allowance, and requires the two leaves to import nothing but
the standard library rather than allow-listing them by name. The same file checks what the
interpreter actually loaded, and that **only `pipeline.py` invokes a stage** — AD-4 says no caller
may invoke a stage out of band, and calling `run` or `run_at_ceiling` is how a caller would.

Which modules may import a stage at all is a separate, exact list in that file, each entry carrying
its reason. `pipeline.py` is one. The other is `corpus/dressings.py`, which reads
`invisible.ZERO_WIDTH` and `invisible.REMOVED` as the character source for the zero-width dressing:
story 3.4 requires the dressing generator and the layer to share one character source, and a second
hand-list of invisible characters is exactly what that rule exists to prevent. Reading a stage's
declared data reorders nothing; calling its entry point is what AD-4 forbids, and that is the
property the test now checks directly rather than through the import that used to stand in for it.
"""
