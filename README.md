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
- [ ] "what this does not show"

## License

MIT. See [LICENSE](LICENSE).
