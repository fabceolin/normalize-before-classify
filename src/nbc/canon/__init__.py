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
interpreter actually loaded, and that only `pipeline.py` imports the stage modules — AD-4 says no
caller may invoke a stage out of band, and an import is how a caller would get the chance.
"""
