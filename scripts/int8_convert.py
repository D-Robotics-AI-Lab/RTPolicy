#!/usr/bin/env python3
"""INT8 conversion is intentionally unsupported for RTPolicy."""


def main():
    raise SystemExit(
        "INT8 is not supported because of observed accuracy degradation. "
        "Use scripts/fp16_convert.py for NVIDIA TensorRT FP16. "
        "No model, calibration data, or output has been modified."
    )


if __name__ == "__main__":
    main()
