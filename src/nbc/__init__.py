"""normalize-before-classify: how much recall does canonicalizing the input first recover?

Deliberately empty of imports. `import nbc` must cost nothing and must not pull in
`onnxruntime`: the platform preflight has to run before that import, or the floor it checks
is a floor the import already crashed through.

The layout is the project's design, in five namespaces:

- `canon/`     the canonicalization layer: ordered, pure text-to-text filters
- `corpus/`    the only writer of `data/*.jsonl`
- `baselines/` the model boundary: one port, one ONNX adapter, CPU only
- `harness/`   the entrypoint, measurement, timing, aggregation, verdicts
- `report/`    renders from `results.json` and reads nothing else

plus three leaf modules: `schema.py` (every record type), `pins.py` (every remote artifact),
`platform.py` (the reproduction floor), and `errors.py` (every abort).
"""
