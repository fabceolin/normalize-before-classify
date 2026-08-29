# Spikes

Exploratory scripts. **Nothing in here is part of the published measurement path**, nothing under
`src/nbc/` imports from here, and no result in `README.md` is produced by anything in this
directory. A spike answers a question that has to be answered *before* something expensive is
built, and it is kept afterwards only as the record of how that question was answered.

A spike may reach the network. The unit suite may not, and no test imports a spike's `main`.

## `oq2_clean_recall.py`

Answers the PRD's open question OQ2: *is each pinned baseline strong enough on clean text for its
encoded degradation to mean anything?* A baseline whose clean recall is already poor makes its own
degradation meaningless, and a baseline that fails OQ2 must be **replaced, never removed** — SC5's
floor is two baselines and the run sits exactly on it.

It reads attack-positive rows straight from the pinned attack dataset (no corpus exists yet), and
scores them through the same model boundary the published run uses — `nbc.baselines.open_windower`
and `nbc.baselines.open_baseline` — so a baseline is never failed by the spike's own tokenization,
windowing or softmax.

```
# pyarrow comes with the already-declared build extra:
uv sync --frozen --extra build
uv run python spikes/oq2_clean_recall.py            # every attack positive, both splits
uv run python spikes/oq2_clean_recall.py --limit 400
```

The outcome of the run, and the date and revision it was decided against, are recorded in
`pins.toml` under each baseline's `[baseline.oq2]` block, where `pins.py` refuses a result that was
measured against a revision the file no longer pins.

**Re-measuring after a pin moves.** That block is required, so a moved pin makes `pins.toml`
unloadable until the OQ2 record names the new revision — and the spike is what produces the number
that record carries. Do not edit the committed file into a state where it declares a recall it
does not have: copy it, point the copy at the new revision, and measure against the copy.

```
mkdir -p /tmp/oq2 && cp pins.toml /tmp/oq2/     # then edit the copy's decided_revision
uv run python spikes/oq2_clean_recall.py --pins-root /tmp/oq2
```

Only then is the committed `pins.toml` updated, with the measurement and the revision together.
