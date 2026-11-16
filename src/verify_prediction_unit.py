from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import service.predict_service as service
from streamlit.testing.v1 import AppTest


def request_json(url: str, method: str = "GET", payload: dict | None = None) -> tuple[int, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body


def assert_no_app_errors(app_test: AppTest, page_name: str) -> None:
    if app_test.exception:
        details = "; ".join(str(item.value) for item in app_test.exception)
        raise AssertionError(f"Streamlit page {page_name!r} raised an exception: {details}")


def main() -> None:
    config = service.load_config()
    if float(config.get("service_input_scale_factor", -1)) != 1.0:
        raise AssertionError("service_input_scale_factor must be 1.0")
    if config.get("model_input_unit") != "%":
        raise AssertionError("model_input_unit must be percent")

    sample = pd.read_csv(ROOT / "outputs" / "sample_input.csv").head(2)
    first = sample.iloc[0].to_dict()
    feature_frame = service._feature_frame(first)
    for name in config["numeric_features"]:
        if name in feature_frame.columns and not np.isclose(
            float(feature_frame.iloc[0][name]), float(first[name]), rtol=0, atol=1e-12
        ):
            raise AssertionError(f"Service changed percentage input for {name}")

    direct = service.predict_record(first, save=False)
    direct_value = float(direct["predicted_yield_strength_MPa"])
    batch = service.predict_dataframe(sample)
    batch_value = float(batch.iloc[0]["predicted_yield_strength_MPa"])
    if "Yield Strength (MPa)" in batch.columns:
        raise AssertionError("Unlabelled demo input unexpectedly contains a target")

    # A second multiplication would be an invalid 0–100 input now.  Verify
    # that the service rejects it instead of silently changing the model
    # input scale.
    double_scaled_input = dict(first)
    for name in config["numeric_features"]:
        double_scaled_input[name] = float(double_scaled_input[name]) * 100.0
    try:
        service.predict_record(double_scaled_input, save=False)
    except ValueError as exc:
        double_scaled_error = str(exc)
    else:
        raise AssertionError("A double-scaled 0–100 input was not rejected")

    # Exercise each Streamlit route without creating or saving screenshot files.
    app_test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert_no_app_errors(app_test, "initial")
    if not app_test.radio:
        raise AssertionError("Streamlit page selector was not rendered")
    page_names = list(app_test.radio[0].options)
    page_results: dict[str, str] = {}
    for page_name in page_names:
        app_test.radio[0].set_value(page_name).run()
        assert_no_app_errors(app_test, page_name)
        page_results[page_name] = "rendered without exception"
    about_page = "关于与说明"
    app_test.radio[0].set_value(about_page).run()
    assert_no_app_errors(app_test, about_page)
    font_info = "中文字体自动探测并回退"
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    if font_info not in app_source:
        raise AssertionError("About page does not include the required font fallback note")

    single_page = "单条预测"
    if single_page not in page_names:
        raise AssertionError(f"Expected Streamlit page not found: {single_page}")
    app_test.radio[0].set_value(single_page).run()
    assert_no_app_errors(app_test, single_page)
    processing_selector = next(
        (item for item in app_test.selectbox if item.label == "Processing 状态"), None
    )
    if processing_selector is None:
        raise AssertionError("Single prediction Processing selector was not rendered")
    processing_selector.set_value(first[config["processing_field"]])
    input_by_name = {
        item.label.removesuffix(" (%)"): item for item in app_test.number_input
    }
    if set(input_by_name) != set(config["numeric_features"]):
        raise AssertionError("Single prediction page fields differ from feature_config.json")
    for name, widget in input_by_name.items():
        widget.set_value(float(first[name]))
    rendered_input_values = {
        item.label.removesuffix(" (%)"): float(item.value) for item in app_test.number_input
    }
    rendered_processing = next(
        item.value for item in app_test.selectbox if item.label == "Processing 状态"
    )
    submit = next((item for item in app_test.button if item.label == "提交"), None)
    if submit is None:
        raise AssertionError("Single prediction submit button was not rendered")
    submit.click().run()
    assert_no_app_errors(app_test, single_page)
    success_messages = [str(item.value) for item in app_test.success]
    match = next((re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MPa", item) for item in success_messages if "MPa" in item), None)
    if match is None:
        raise AssertionError(f"Single prediction page produced no success result: {success_messages}")
    web_value = float(match.group(1))
    if not np.isclose(web_value, direct_value, rtol=0, atol=0.0051):
        service.clear_records()
        raise AssertionError(
            f"Web page and direct service predictions differ beyond display rounding: "
            f"web={web_value}, service={direct_value}, messages={success_messages}, "
            f"input={rendered_input_values}, processing={rendered_processing}"
        )
    streamlit_records_count = len(service.list_records(limit=100000))
    if streamlit_records_count != 1:
        service.clear_records()
        raise AssertionError(
            f"Streamlit prediction did not save exactly one record: {streamlit_records_count}"
        )
    service.clear_records()

    missing_payload = dict(first)
    missing_payload.pop(config["numeric_features"][0])
    try:
        service.predict_record(missing_payload, save=False)
    except (ValueError, KeyError) as exc:
        missing_error = str(exc)
    else:
        raise AssertionError("Missing input field was not rejected")
    invalid_payload = dict(first)
    invalid_payload[config["numeric_features"][0]] = "not-a-number"
    try:
        service.predict_record(invalid_payload, save=False)
    except (ValueError, KeyError) as exc:
        invalid_error = str(exc)
    else:
        raise AssertionError("Non-numeric input was not rejected")

    service.clear_records()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "error"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health_status = None
        health_result: object = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"uvicorn exited with code {server.returncode}")
            try:
                health_status, health_result = request_json(base_url + "/health")
                break
            except (URLError, TimeoutError, OSError):
                time.sleep(0.25)
        if health_status != 200:
            raise RuntimeError("API did not become ready on the local test port")

        api_status, api_result = request_json(base_url + "/predict", "POST", first)
        if api_status != 200 or not isinstance(api_result, dict):
            raise AssertionError(f"API /predict failed: HTTP {api_status}: {api_result}")
        api_value = float(api_result["predicted_yield_strength_MPa"])
        records_status, records_result = request_json(base_url + "/records")
        invalid_status, _ = request_json(base_url + "/predict", "POST", {})
        if records_status != 200 or not isinstance(records_result, list):
            raise AssertionError("API /records did not return a JSON list")
        if invalid_status != 422:
            raise AssertionError(f"Invalid API input returned HTTP {invalid_status}, expected 422")
        if not isinstance(health_result, dict) or health_result.get("status") != "ok":
            raise AssertionError(f"API /health returned an unexpected body: {health_result}")
        if not np.isclose(direct_value, api_value, rtol=0, atol=1e-12):
            raise AssertionError("Direct service and HTTP API predictions differ")
        if len(records_result) != 1:
            raise AssertionError(f"Expected one HTTP API test record after isolating its DB, got {len(records_result)}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=8)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=8)
        service.clear_records()

    remaining_records = len(service.list_records(limit=100000))
    if remaining_records != 0:
        raise AssertionError("records.db was not cleared after the regression test")
    values = [direct_value, batch_value, api_value]
    if max(values) - min(values) != 0.0:
        raise AssertionError("Service, batch, and API prediction values are not identical")

    result = {
        "sample_input_path": "outputs/sample_input.csv",
        "sample_processing": first[config["processing_field"]],
        "sample_Al_percent": float(first["Al"]),
        "service_input_scale_factor": config["service_input_scale_factor"],
        "model_input_unit": config["model_input_unit"],
        "double_scaled_input_rejected": True,
        "double_scaled_input_error": double_scaled_error,
        "direct_service_prediction_MPa": direct_value,
        "batch_service_prediction_MPa": batch_value,
        "streamlit_single_page_prediction_display_MPa": web_value,
        "api_http_prediction_MPa": api_value,
        "prediction_max_difference_MPa": max(values) - min(values),
        "web_display_rounding_difference_MPa": abs(web_value - direct_value),
        "unlabelled_batch_rows": len(batch),
        "unlabelled_batch_prediction_column_present": "predicted_yield_strength_MPa" in batch.columns,
        "streamlit_page_count": len(page_names),
        "streamlit_pages_tested": page_results,
        "streamlit_font_caption": f"关于与说明/环境：{font_info}",
        "streamlit_prediction_records_saved": streamlit_records_count,
        "missing_input_error": missing_error,
        "non_numeric_input_error": invalid_error,
        "api_http_health_status": health_status,
        "api_http_predict_status": api_status,
        "api_http_records_status": records_status,
        "api_http_invalid_predict_status": invalid_status,
        "api_http_records_during_test": len(records_result),
        "records_db_rows_after_clear": remaining_records,
    }
    output = ROOT / "artifacts" / "prediction_unit_regression.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
