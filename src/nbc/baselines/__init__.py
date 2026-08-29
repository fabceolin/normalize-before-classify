"""The model boundary: one port, one ONNX adapter, CPU only.

Every session names `providers=["CPUExecutionProvider"]` explicitly and reads no device from
the environment, so a difference between two columns is a difference between two models
rather than between two code paths or two devices.
"""
