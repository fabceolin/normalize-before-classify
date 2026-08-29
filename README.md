# normalize-before-classify

**Measuring what input canonicalization buys a prompt-injection classifier, and what it costs in false
positives.**

> 🚧 Work in progress, started 2026-08-22. The numbers below do not exist yet. This README states the
> question before the answer is known, on purpose: the measurement decides the claim, not the other way
> around.

## The question

Prompt-injection classifiers are trained on attack text. Attackers stopped sending attack text. A payload
wrapped in base64, split with zero-width characters, or spelled with Cyrillic homoglyphs is, to a
tokenizer, a different string than the attack it carries.

There are two broad answers: teach the model to read the encodings, with another fine-tuning round per
encoding family, or **canonicalize the input before the model sees it** — NFKC, strip zero-width and bidi
overrides, map confusables, detect and decode embedded base64 and hex with a declared depth ceiling. Linear
in the length of the text, no retraining.

**Normalizing input before classification is not a new idea, and this repository does not claim it is.** It
is established practice, and the version of this argument that paints the field as reflexively reaching for
another fine-tune is a caricature. What is missing is not the idea but the measurement: what the layer
actually recovers, what it actually costs, with counts and intervals, on baselines that can disagree with
each other, reproducible from one command. That is what this repository is.

## What gets measured

Two halves, because only reporting the first half would be dishonest. Every rate carries its item count and
a 95% interval, and the false-positive rate is reported **per benign class**, never pooled:

| | recall on encoded attacks | false positives, code | false positives, chat |
|---|---|---|---|
| baseline classifier | | | |
| baseline + canonicalization | | | |

## Reproducing this

The published run is one command, and it does not exist yet: the entrypoint that performs the whole
sequence arrives with the measurement harness. Stating it here before it runs would be a reproduction
claim nobody can act on, which is the defect the rest of this file exists to refuse.

What does run today, on a clean CPU-only Linux machine:

```
uv sync --frozen --extra build --group dev
uv run python -m nbc.platform    # the platform floor, checked before anything else
uv run python -m nbc.pins --verify   # every pinned artifact, resolved and checked
uv run pytest                    # the offline unit suite, no network, no model download
```

`uv` is pinned to an exact version by `pyproject.toml` and refuses to run under any other, so the
environment a table came from is the environment you get. The models and the corpus are pinned by
revision in `pins.toml`, and `nbc.pins --verify` reports whether each one was checked against the hub or read
off a directory name in this machine's cache — those are different guarantees and it says which.

Linux specifically, and the reason is stated rather than implied: the pinned `onnxruntime`
publishes only `manylinux_2_28` wheels and no source distribution at any version, so the floor is
**glibc 2.28 or newer** on `x86_64` or `aarch64`. The interpreter is **CPython 3.13 exactly** —
not the 3.11-to-3.14 the wheels would admit — because the vendored Unicode confusables table is
pinned to a revision that moves with the interpreter's minor version. All of that is checked
before anything else runs, and it names what it found:

```
uv run python -m nbc.platform
```

The table above is the shape, hand-drawn and empty. The filled one is generated from `results/results.json` and
injected between the two markers below, which are empty because no run has produced a table yet. Nothing
between them is ever written or edited by hand, so a number in the table that no run produced cannot
exist.

<!-- RESULTS:START -->
<!-- RESULTS:END -->

Repeated for each dressing (clean, base64, hex, homoglyph, zero-width) and for each of two public
baselines, pinned by revision, chosen to span two architectures and two tokenizer families — because the
mechanism under suspicion is how encoded text tokenizes, and models that tokenize alike cannot corroborate
each other. Two is the minimum this claim can rest on, not a comfortable margin; a third was pinned and
then dropped, and the reason is in "what this does not show" rather than in a commit message.

One thing to check before you check it yourself: the second baseline's model card declares a DeBERTa base
model in its metadata, which is boilerplate inherited from a family card and is wrong. The pinned
revision's own `config.json` is what was verified — it is a BERT with a 30k WordPiece vocabulary, against
the first baseline's DeBERTa-v3 with a 128k SentencePiece vocabulary. The independence is real; one of the
two cards says it is not.

There is also a second block of the table built from encodings the layer was deliberately never written
against. That block is the one that could come out badly, and it is where the pre-registered pass/fail
conditions are decided.

The benign corpus matters as much as the attack corpus. JWTs, data URIs, hashes, SSH keys and base64 blobs
inside source code are ordinary traffic. A canonicalizer that decodes everything it finds will turn a
recall win into a false-positive problem, and any guardrail team that has shipped a classifier over coding
traffic already knows that pain. Code and chat are reported separately because a layer that is safe on chat
and destructive on code looks acceptable in a pooled number, and that is precisely the failure worth
knowing about.

Also reported: the wall-clock cost of the canonicalization layer itself, per document, at p50 and p95,
separately from inference time. No figure for it is claimed here — that is what the run is for.

## Status

- [ ] attack corpus from pinned public datasets, in five dressings: clean, base64, hex, homoglyph, zero-width
- [ ] benign corpus, two classes reported separately: real pinned public source files, and conversational
      text carrying legitimate encoded content
- [ ] canonicalization layer with a declared recursion ceiling
- [ ] measurement harness and results table, every rate with its n and interval
- [x] "what this does not show" — the eleven caveats that do not depend on the result, written
      before the first run; slot 8 waits for what the run reveals

## How big the layer is

The whole argument here is that a small, readable canonicalization layer buys most of what another
fine-tuning round would. "Small" and "readable" are adjectives, and a repository that pins a Unicode
revision and a glibc minor should not be asking anyone to take an adjective on trust. So the layer
carries a stated budget, in lines, over the modules that actually run in front of a classifier:

<!-- SIZE-BUDGET:START -->
- `total_physical_lines`: 2000
- `total_code_lines`: 1000
- `module_physical_lines`: 550
<!-- SIZE-BUDGET:END -->

`total_physical_lines` is every line a reviewer scrolls. `total_code_lines` counts only the lines
that are neither blank, nor comments, nor docstrings — the part that has to be reasoned about. The
third is a ceiling on any single module, because "read it end to end" degrades far faster with one
1500-line file than with five 300-line ones. The build-time script that derives the vendored
confusables table is excluded and the exclusion is checked, not asserted: it must be exactly the set
of files a real import of the layer never reaches.

    python -m nbc.report.size_budget

prints the measurement, the budget and the headroom, and aborts with its own exit code if the layer
outgrew the budget or if the numbers above stopped matching the ones the code declares. Editing any
one of the three without the other two fails.

**What the budget does not prove.** It is a ceiling on growth, not evidence that the layer reads in
one sitting: a ceiling in the low thousands of lines is a couple of hours of careful reading, not
twenty minutes. No measurement is transcribed into this paragraph on purpose — a number written here
would go stale the next time a docstring changes, and a stale number beside a checked one is worse
than no number. Run the command above for where the layer actually stands. The point is that this is
now a budget to disagree with rather than a claim the repository makes about itself.

## What this does not show

These eleven do not depend on the result, so they were written before the first measurement rather than
after it, and the numbering is the PRD's — a reader moving between the two documents lands on the same
caveat. The section is hand-written and sits outside the generated block above; slot 8 is reserved for
what the run actually reveals. The run will not start without it: `python -m nbc.report.caveats` checks
that the section is here and complete, and aborts with its own exit code when it is not.

**1. The corpus is constructed, not sampled from production traffic.** Attack payloads come from public,
revision-pinned datasets and are re-rendered here in each dressing. The benign corpus is drawn the same
way — real public source files pinned by repository, commit and path for the code half, and the
benign-labelled rows of the same pinned datasets for the conversational half — under a sampling frame
fixed and recorded before any measurement. Hand-authored material is confined to what no public dataset
carries: messages legitimately containing a JWT, a content hash, a data URI or an SSH public key. Nothing
here is a sample of what real traffic looks like, and the rates below should be read as a comparison
between conditions, not as an estimate of how often either failure occurs in the wild. One consequence
worth naming: the code half is drawn from a bounded list of repositories, and files from one repository
resemble each other, so the intervals on that half are somewhat narrower than fully independent sampling
would justify. The repository count is reported so a reader can judge by how much.

**1b. Language is not a controlled variable.** The pinned attack dataset reads as predominantly English
in sample but was never audited for language, and neither were the datasets its own card says it was
seeded from. Nothing filters by language and nothing here claims to. A result that differs across
languages would be invisible in this table.

**2. The classifier is treated as a black box.** Both baselines are accessed through their public
inference interface, with no access to activations or internal state. A whole class of defenses works the
other way, reading the model's internal state rather than classifying the input text, and commercial
systems built on that approach exist. This result says nothing about them, in either direction.

**3. There are two baselines, which is the minimum this claim can rest on, and one of them is obscure.**
They were chosen for independence rather than popularity: two architectures (DeBERTa-v3, BERT) and two
tokenizer families (SentencePiece 128k, WordPiece 30k), because the mechanism under suspicion is how
encoded text tokenizes, and models that tokenize alike cannot corroborate each other. Only the first is
downloaded at scale. **A third baseline was pinned and then dropped**, and the reason belongs here rather
than in a commit message: its published training sources included the attack datasets used here, and its
training augmentations were the same encodings this experiment applies — so its recall on encoded payloads
would have measured robustness it had been trained for, not the effect under study. Dropping it also cost
one of the two attack datasets, which a second baseline had likewise trained on. Two independent
baselines is a floor, not a comfortable margin, and a reader is entitled to weigh the result accordingly.

**3b. Both baselines cap sequences at the same length**, so long documents are windowed for both and the
per-baseline share of items that exceeded one window is reported alongside the rates. That symmetry is
convenient here and is not a property of the model class: the dropped third baseline had a context four
times longer and published its own long-document protocol, and had it stayed, its column would have been
produced under a different windowing regime from the others. The comparison in this table does not have
that problem; a reader extending it to other models should not assume the same.

**3c. This measures whether a classifier fires, not whether an attack works.** A payload the classifier
misses is a recall failure by construction, but it is only a *threat* if the model downstream decodes it
and obeys. Nothing here tests that. If some encodings are inert against a downstream model anyway, the
recall recovered on them is worth less than the table makes it look; if they are not inert, the recall
lost to them is worth more. Establishing which would need a downstream model and a definition of attack
success, and that is a different experiment. Read the rates as a comparison between conditions at the
classifier, and nothing further.

**3d. Training-data overlap is filtered where it can be measured, and one source cannot be.** A classifier
scored on text it was trained on reports memory, not detection — and it cuts both ways: recall looks better
on attacks it has seen, and the false-positive rate looks better on benign text it was taught to call safe.
Reading model cards is not enough to catch this. The attack corpus used here declares on its own card that
it was seeded from two datasets that one of the baselines declares training on, one hop that no model card
reveals; measured literally, that reached **48% of the benign rows and 17% of the attack rows** before
filtering. Read those two figures as a floor rather than as the removal: they were measured against those
two seeds alone, and the exclusion set the build actually applies is wider — every source either baseline
declares training on, plus every seed the attack dataset's own card names, twelve today, each pinned in
`pins.toml` by repository and — where the hub will resolve one — revision, and derived from those
declarations rather than listed beside them,
so a lineage that grows and an exclusion set that does not is a file that no longer loads. Two texts are
the same row under a declared normalization: NFKC, lowercased, whitespace collapsed. So the build downloads
every declared training source it can and removes every corpus row that appears in one, and reports how
many rows each source removed. What remains undeclared or unreachable remains a limit, and there are two
of those. One training source common to **both** baselines is access-restricted and returns HTTP 401, so
overlap against it is unmeasurable for any corpus this experiment could have chosen. A second — one of the
four seeds the attack dataset names — resolves at its pinned commit and publishes its rows through a
loading script the pinned reader refuses, so its overlap is unmeasurable too, for a different reason and
with a different remedy. Both are stated here rather than quietly assumed to be zero: each is named in the
results with the status it returned and the refusal it raised, its removal count is reported as absent
rather than as zero, and the pins record which of the two failures it is. Neither is one of the two seeds
the declared one-hop reach runs through — those two the build may not proceed without, and it aborts if it
cannot read them. And that one source is a floor on what is missing, not a
full accounting: the primary baseline's card names far fewer training datasets in prose than its own
metadata tallies, so the number of sources this filter never sees is itself unknown. What the filter
reaches is what the cards name and what can be downloaded, which is not the same as what these models
were trained on.

**4. No adaptive evasion was attempted.** Every payload here was encoded without knowledge of the
canonicalization layer. An attacker who reads this repository is a different adversary, and defeating a
published normalizer is a materially easier problem than defeating an unknown one. This is the most
important follow-up.

**5. Long documents are windowed, and a window boundary can split an instruction.** Documents past a
model's sequence limit are scored in fixed non-overlapping windows and the document takes the maximum
window score. An instruction that straddles a boundary is seen by neither window in full. The direction of
this bias is worth stating, because it is the opposite of what one might assume: canonicalization
*shortens* documents in tokens rather than lengthening them — an encoded payload tokenizes into several
times the tokens of its decoded form — so it is the **encoded** condition that spills into extra windows,
and taking the maximum across windows hands those extra chances to the un-canonicalized baseline.
On the recall side, that works against the result reported here. **It does not work the same way on the
other half, and stating only the flattering half would be the kind of selective candour this section
exists to avoid.** Benign items are dressed in the same encodings, so encoded benign documents also spill
into extra windows without the layer, also take an inflated maximum, and also produce extra false
positives without the layer — which makes the layer's false-positive *cost* look smaller than it is. The
two biases push the headline comparison in opposite directions and do not cancel in any knowable way.
Both are reported: the share of items needing more than one window is given per baseline, and every
cost-versus-gain comparison is accompanied by a **windows-matched** version restricted to documents that
fit in a single window under both conditions, so a reader can see how much of the trade is the window
policy rather than the layer.

**6. The layer is itself a surface.** A canonicalizer that decodes aggressively can be attacked from the
other direction: benign-looking encoded content that decodes into text the classifier fires on turns a
guardrail into a denial-of-service against whoever deploys it. The declared recursion ceiling bounds
expansion, not this. Measuring it is future work and is not attempted here.

**7. Part of the recall recovery is true by construction, and you should discount it accordingly.** The
code that encodes the payloads and the layer that decodes them draw on the same character tables, bound by
a test that fails the build if the layer does not undo its own corpus's dressing. That binding exists for a
good reason — without it the layer could silently fail to strip a character it emitted, quietly depressing
the headline number with nothing failing loudly — but it has a consequence worth stating plainly: on the
dressings the layer was written for, the canonicalized encoded document *is* the canonicalized clean
document, so recovery is total by definition and could not have come out any other way. That column shows
the layer was implemented as specified. It is not evidence that canonicalization is a good idea. The
columns that could have gone differently are the recall **lost** without the layer, the false-positive cost
of adding it, the behaviour on nested chains past the recursion ceiling, and the **held-out encodings** —
a separate block of the table built from encodings the layer was deliberately never written against, kept
in their own registry and exempt from the binding above precisely so that something here could come out
badly. Read those. One held-out encoding, `rot13`, is included knowing the layer cannot touch it: it is
there to mark the boundary of what normalization can reach, not to be recovered, and it is reported and
excluded from the pass/fail condition for that reason.

**8.** *(reserved for what the run actually revealed — written after the numbers exist)*

## License

MIT. See [LICENSE](LICENSE).
