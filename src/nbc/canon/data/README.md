# The vendored confusable mapping

`confusables-<revision>.json` is derived from the Unicode Security Mechanisms table for UTS #39
(`confusables.txt`). It is committed so the layer's character mapping cannot change between runs
and needs no network at canonicalization time.

**Vendored at Unicode 15.1.0**, as `confusables-15.1.0.json`, holding 95 mappings. A test asserts
that this file names the revision the artifact actually carries, so a re-vendoring that leaves this
page describing the old table fails rather than misleading the next reader.

## What it is, and what it deliberately is not

- Keys are **single non-ASCII code points** in Cyrillic and Greek. Values are **ASCII** strings.
  The mapping is applied per code point and is the **identity on all of `U+0000..U+007F`**.
- It is **not** a UTS-39 skeleton. The upstream table carries
  `0031 ; 006C ; MA # ( 1 → l ) DIGIT ONE → LATIN SMALL LETTER L`, and seven more rows with ASCII
  sources. A skeleton transform keeps them, folds `1` to `l` and `0` to `O` across ordinary source
  code, and would do two things this experiment cannot survive: turn the benign-code
  counter-metric into a number about ASCII folding, and corrupt base64 and hex runs before the
  decode stage ever saw them.
- The upstream prototype is taken **as-is**. UTS-39's table is transitively closed, so an in-scope
  code point whose prototype is not ASCII has no ASCII form at all and is dropped rather than
  chased. A `smoke`-marked test checks that closure against the live table.

## The revision is pinned to the interpreter

The revision is in the **filename**, and `nbc.canon.confusables_table.load()` refuses unless it
equals `unicodedata.unidata_version`. Step 2 of the pipeline is this mapping and step 3 is NFKC,
which is the interpreter's own tables; at two different revisions they can disagree about the same
character with no symptom other than a number that moved. That is the whole reason `pyproject.toml`
pins CPython 3.13 exactly while the `onnxruntime` wheels would admit 3.11 through 3.14.

**A Python minor bump means re-vendoring and a full re-run, not a test update.**

## Reproducing it

    python -m nbc.canon.vendor_confusables --check    # re-derive from upstream and byte-compare
    python -m nbc.canon.vendor_confusables --write    # re-vendor (delete the old artifact first)

The derivation is `src/nbc/canon/vendor_confusables.py`, and its rules are tested offline against a
fixture in the real upstream format. The artifact records the URL, SHA-256 and byte count of the
upstream file it came from; the `smoke` tier fetches that file and compares all three, then
re-derives the mapping and compares that too.

## Upstream notice

    © 2023 Unicode®, Inc.
    For terms of use, see https://www.unicode.org/terms_of_use.html
