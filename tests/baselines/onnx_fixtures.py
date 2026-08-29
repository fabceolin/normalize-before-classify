"""Real ONNX graphs, built in-process, so the model boundary is tested with no model download.

The claims this story makes -- the session runs on `CPUExecutionProvider`, the feed is built
from the graph's *declared* signature, batching changes the float and not the class -- are
claims about `onnxruntime` behaviour. Substituting a fake session for the real one would test
the double instead: `get_providers()` would return whatever the fake was told to return, and
the signature the feed is built from would be the signature the test wrote by hand at both ends.

So the tests use a real session over a real graph. The graph is emitted here, as ONNX protobuf
bytes, from a few hundred bytes of hand-written wire format -- ONNX ships no writer in this
project's dependency set and adding one would put a package in the resolution that the
published run never executes.

The graphs are deliberately trivial arithmetic over the token ids. What matters about them is
their **shape**: which inputs they declare, with which types, and whether the number they emit
moves when a batch pads. `reduce="mean"` is the one that moves, which is what makes the
batch-invariance test a test rather than a tautology.
"""

from __future__ import annotations

from typing import Final, Sequence

# ONNX TensorProto data types, by their numbers in the ONNX spec.
FLOAT: Final[int] = 1
INT32: Final[int] = 6
INT64: Final[int] = 7

# ONNX AttributeProto.AttributeType.
_ATTR_INT: Final[int] = 2
_ATTR_INTS: Final[int] = 7

_IR_VERSION: Final[int] = 7
_OPSET: Final[int] = 11
"""Opset 11 keeps `ReduceSum`'s axes an attribute, so no graph here needs an initializer."""


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return bytes(out)


def _tag(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def _int_field(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(value)


def _bytes_field(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _string_field(field: int, text: str) -> bytes:
    return _bytes_field(field, text.encode("utf-8"))


def _value_info(name: str, elem_type: int, dims: Sequence[int | str]) -> bytes:
    """A `ValueInfoProto`: a name plus a tensor type with a shape."""
    shape = b"".join(
        _bytes_field(1, _int_field(1, dim) if isinstance(dim, int) else _string_field(2, dim))
        for dim in dims
    )
    tensor_type = _int_field(1, elem_type) + _bytes_field(2, shape)
    return _string_field(1, name) + _bytes_field(2, _bytes_field(1, tensor_type))


def _attribute_int(name: str, value: int) -> bytes:
    return _string_field(1, name) + _int_field(3, value) + _int_field(20, _ATTR_INT)


def _attribute_ints(name: str, values: Sequence[int]) -> bytes:
    packed = b"".join(_varint(value) for value in values)
    return _string_field(1, name) + _bytes_field(8, packed) + _int_field(20, _ATTR_INTS)


def _node(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str,
    attributes: Sequence[bytes] = (),
) -> bytes:
    return (
        b"".join(_string_field(1, value) for value in inputs)
        + b"".join(_string_field(2, value) for value in outputs)
        + _string_field(3, name)
        + _string_field(4, op_type)
        + b"".join(_bytes_field(5, attribute) for attribute in attributes)
    )


def classifier_graph(
    *,
    inputs: Sequence[str] = ("input_ids", "attention_mask"),
    input_type: int = INT64,
    num_labels: int = 2,
    reduce: str = "sum",
    rank: int = 2,
) -> bytes:
    """A graph shaped like a sequence classifier: `[batch, sequence]` in, `[batch, labels]` out.

    Every declared input is reduced along the sequence axis and the reductions are added, so no
    input is dead -- a graph whose inputs the runtime could prune would not be a signature to
    build a feed from. The logit for label 0 is the negation of the rest, so a positive token
    sum puts the argmax on the last label and a caller can predict the class it should see.

    `reduce="mean"` divides by the padded width, which is exactly the sensitivity to batching
    that AD-22 says must not change a class.
    """
    if not inputs:
        raise ValueError("a classifier graph declares at least one input")
    if num_labels < 2:
        raise ValueError("a classifier graph declares at least two labels")

    axis = 1 if rank == 2 else 0
    operator = {"sum": "ReduceSum", "mean": "ReduceMean"}[reduce]

    nodes: list[bytes] = []
    reduced: list[str] = []
    for position, name in enumerate(inputs):
        nodes.append(
            _node("Cast", [name], [f"float_{position}"], f"cast_{position}",
                  [_attribute_int("to", FLOAT)])
        )
        nodes.append(
            _node(
                operator,
                [f"float_{position}"],
                [f"reduced_{position}"],
                f"reduce_{position}",
                [_attribute_ints("axes", [axis]), _attribute_int("keepdims", 1)],
            )
        )
        reduced.append(f"reduced_{position}")

    total = reduced[0]
    for position, name in enumerate(reduced[1:], start=1):
        nodes.append(_node("Add", [total, name], [f"sum_{position}"], f"add_{position}"))
        total = f"sum_{position}"

    nodes.append(_node("Neg", [total], ["negated"], "neg"))
    nodes.append(
        _node(
            "Concat",
            ["negated"] + [total] * (num_labels - 1),
            ["logits"],
            "concat",
            [_attribute_int("axis", axis)],
        )
    )

    in_dims: tuple[int | str, ...] = ("batch", "sequence") if rank == 2 else ("sequence",)
    out_dims: tuple[int | str, ...] = ("batch", num_labels) if rank == 2 else (num_labels,)
    graph = (
        b"".join(_bytes_field(1, node) for node in nodes)
        + _string_field(2, "nbc-test-classifier")
        + b"".join(_bytes_field(11, _value_info(name, input_type, in_dims)) for name in inputs)
        + _bytes_field(12, _value_info("logits", FLOAT, out_dims))
    )
    return (
        _int_field(1, _IR_VERSION)
        + _string_field(2, "nbc-tests")
        + _bytes_field(7, graph)
        + _bytes_field(8, _string_field(1, "") + _int_field(2, _OPSET))
    )
