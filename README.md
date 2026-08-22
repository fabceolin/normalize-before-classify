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

There are two ways to answer that:

1. **Teach the model to read the encodings.** Another fine-tuning round, for every encoding, forever.
2. **Canonicalize the input before the model sees it.** NFKC, strip zero-width and bidi overrides, map
   confusables, detect and decode embedded base64 and hex with a declared depth ceiling. Linear in the
   length of the text, microseconds per document, no retraining.

This repository measures the second option against a public baseline classifier, and it measures the price
of it, which is the part these discussions usually skip.

## What gets measured

Two halves, because only reporting the first half would be dishonest:

| | recall on encoded attacks | false positives on benign encoded text |
|---|---|---|
| baseline classifier | | |
| baseline + canonicalization | | |

The benign corpus matters as much as the attack corpus. JWTs, data URIs, hashes, SSH keys and base64 blobs
inside source code are ordinary traffic. A canonicalizer that decodes everything it finds will turn a
recall win into a false-positive problem, and any guardrail team that has shipped a classifier over coding
traffic already knows that pain.

Also reported: the cost of the canonicalization layer itself, in microseconds per document.

## Status

- [ ] attack corpus (public, canonical payloads only) in five dressings: clean, base64, hex, homoglyph, zero-width
- [ ] benign corpus with legitimate encoded content
- [ ] canonicalization layer with a declared recursion ceiling
- [ ] measurement harness and results table
- [ ] "what this does not show"

## License

MIT. See [LICENSE](LICENSE).
