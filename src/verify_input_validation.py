from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service import predict_service as service


def main() -> None:
    config = service.load_config()
    sample = pd.read_csv(ROOT / "outputs" / "sample_input.csv").iloc[0].to_dict()
    previous_output = pd.read_csv(ROOT / "outputs" / "predictions_demo.csv")
    reference = float(previous_output.iloc[0]["predicted_yield_strength_MPa"])
    current = service.predict_record(sample, save=False)
    prediction = float(current["predicted_yield_strength_MPa"])
    if not np.isclose(prediction, reference, rtol=0, atol=1e-12):
        raise AssertionError(f"Sample prediction changed: {reference} -> {prediction}")

    out_of_range = dict(sample)
    out_of_range[config["numeric_features"][0]] = 100.001
    try:
        service.predict_record(out_of_range, save=False)
    except ValueError as exc:
        range_message = str(exc)
        if "0–100%" not in range_message:
            raise AssertionError(f"Unclear range error: {range_message}") from exc
    else:
        raise AssertionError("Out-of-range composition was accepted")

    off_sum = dict(sample)
    off_sum["Al"] = float(off_sum["Al"]) - 5.0
    warning_result = service.predict_record(off_sum, save=False)
    warnings = warning_result.get("warnings", [])
    if not warnings or not np.isfinite(warning_result["predicted_yield_strength_MPa"]):
        raise AssertionError("Off-sum composition did not warn and continue prediction")

    unknown = dict(sample)
    unknown[config["processing_field"]] = "Unknown processing test value"
    try:
        service.predict_record(unknown, save=False)
    except ValueError as exc:
        processing_message = str(exc)
        if "未知 Processing 值" not in processing_message:
            raise AssertionError(f"Unclear Processing error: {processing_message}") from exc
        missing_allowed = [value for value in config["processing_values"] if value not in processing_message]
        if missing_allowed:
            raise AssertionError(f"Allowed Processing values omitted: {missing_allowed}") from exc
    else:
        raise AssertionError("Unknown Processing value was accepted")

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    if app.exception:
        raise AssertionError("Streamlit app failed: " + "; ".join(str(item.value) for item in app.exception))
    page_selector = app.radio[0]
    page_selector.set_value("单条预测").run()
    if app.exception:
        raise AssertionError("Single prediction page failed")
    fill_button = next((item for item in app.button if item.label == "填入典型合金示例"), None)
    if fill_button is None:
        raise AssertionError("Example fill button is missing")
    fill_button.click().run()
    if app.exception:
        raise AssertionError("Example fill interaction failed")
    widgets = {item.label.removesuffix(" (%)"): item for item in app.number_input}
    if set(widgets) != set(config["numeric_features"]):
        raise AssertionError("Single-page numeric inputs do not match feature_config")
    for name in config["numeric_features"]:
        if not np.isclose(float(widgets[name].value), float(sample[name]), rtol=0, atol=1e-12):
            raise AssertionError(f"Example button did not fill {name} from sample_input.csv")
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    app_source = ast.parse(app_text)
    number_inputs = [node for node in ast.walk(app_source) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "number_input"]
    max_values = [keyword.value for call in number_inputs for keyword in call.keywords if keyword.arg == "max_value"]
    option_dict_max_values = [item.value for node in ast.walk(app_source) if isinstance(node, ast.Dict)
                              for key, item in zip(node.keys, node.values)
                              if isinstance(key, ast.Constant) and key.value == "max_value"]
    max_value_literals = [node.value for node in max_values if isinstance(node, ast.Constant)]
    if (not max_values and not option_dict_max_values) or 100.0 not in [*max_value_literals, *option_dict_max_values]:
        raise AssertionError("Streamlit numeric inputs do not declare max_value=100.0")

    result = {
        "sample_prediction_reference_MPa": reference,
        "sample_prediction_after_validation_MPa": prediction,
        "sample_prediction_unchanged": True,
        "out_of_range": {"status": "passed", "message": range_message},
        "off_sum_warning": {"status": "passed", "warnings": warnings,
                            "prediction_MPa": float(warning_result["predicted_yield_strength_MPa"])},
        "unknown_processing": {"status": "passed", "message": processing_message,
                                "allowed_values": config["processing_values"]},
        "streamlit_max_value_100": "passed",
        "example_fill_button": "passed; 25 values match outputs/sample_input.csv row 1",
    }
    output = ROOT / "artifacts" / "input_validation_tests.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
