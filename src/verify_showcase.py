from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from service.comparison_service import comparison_csv, comparison_frame, comparison_markdown
    from service.predict_service import predict_dataframe, predict_record
    from service.sensitivity_service import build_compensated_scan

    cfg = json.loads((ROOT / "feature_config.json").read_text(encoding="utf-8"))
    ranges = json.loads((ROOT / "artifacts" / "sensitivity_ranges.json").read_text(encoding="utf-8"))["ranges"]
    example = pd.read_csv(ROOT / "outputs" / "sample_input.csv").iloc[0]
    base = {name: float(example[name]) for name in cfg["numeric_features"]}
    processing = str(example[cfg["processing_field"]])
    scan_results = {}
    for element in [name for name in cfg["numeric_features"] if name != "Al"]:
        interval = ranges[element]
        scan = build_compensated_scan(base, processing, element, interval["q05_percent"], interval["q95_percent"], 41)
        result = predict_dataframe(scan)
        totals = scan[cfg["numeric_features"]].sum(axis=1)
        scan_results[element] = {
            "points": int(len(scan)),
            "min_total_percent": float(totals.min()),
            "max_total_percent": float(totals.max()),
            "all_totals_equal_100": bool((totals.sub(100.0).abs() <= 1e-9).all()),
            "warnings": list(result.attrs.get("input_warnings", [])),
        }
    direct = predict_record({**base, "Processing": processing}, save=False)
    comparison = comparison_frame([{"record": {**base, "Processing": processing}, "prediction": direct["predicted_yield_strength_MPa"]}], cfg["numeric_features"])
    csv_bytes = comparison_csv(comparison)
    markdown_bytes = comparison_markdown(comparison)
    csv_openable = bool(pd.read_csv(pd.io.common.BytesIO(csv_bytes)).shape[0] == 1)
    markdown_openable = markdown_bytes.startswith(b"# ") and b"\xe9\xa2\x84" in markdown_bytes

    # Exercise the added UI flow without creating or changing screenshots:
    # submit the example on the single-prediction page, render a zero-range
    # element chart, add the exact prediction to comparison, and reopen it.
    app_test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    if app_test.exception:
        raise AssertionError("Initial app render failed: " + "; ".join(str(item.value) for item in app_test.exception))
    page_selector = app_test.radio[0]
    rendered_pages = list(page_selector.options)
    page_selector.set_value("单条预测").run()
    processing_selector = next(item for item in app_test.selectbox if item.label == "Processing 状态")
    processing_selector.set_value(processing)
    for item in app_test.number_input:
        feature = item.label.removesuffix(" (%)")
        item.set_value(base[feature])
    submit_button = next(item for item in app_test.button if item.label == "提交")
    submit_button.click().run()
    if app_test.exception:
        raise AssertionError("Single-prediction page failed: " + "; ".join(str(item.value) for item in app_test.exception))
    sensitivity_selector = next(item for item in app_test.selectbox if item.label.startswith("选择扫描元素"))
    zero_range_element = next(
        (name for name, interval in ranges.items() if name != "Al" and interval["q05_percent"] == interval["q95_percent"]),
        None,
    )
    if zero_range_element is None:
        raise AssertionError("No zero-range element was found for chart-boundary regression")
    sensitivity_selector.set_value(zero_range_element).run()
    if app_test.exception:
        raise AssertionError(
            "Zero-range sensitivity chart failed: "
            + "; ".join(str(item.value) for item in app_test.exception)
        )
    add_button = next(item for item in app_test.button if item.label == "加入对比")
    add_button.click().run()
    page_selector = app_test.radio[0]
    page_selector.set_value("配方对比").run()
    if app_test.exception:
        raise AssertionError("Recipe comparison page failed: " + "; ".join(str(item.value) for item in app_test.exception))
    comparison_values = [
        float(value)
        for dataframe in app_test.dataframe
        for row in dataframe.value.to_dict(orient="records")
        for key, value in row.items()
        if key == "预测屈服强度 (MPa)"
    ]
    ui_comparison_matches = any(np.isclose(value, direct["predicted_yield_strength_MPa"], rtol=0, atol=1e-12) for value in comparison_values)
    if not ui_comparison_matches:
        raise AssertionError("UI comparison page does not show the single-page prediction value")
    comparison_controls = {item.label for item in app_test.get("download_button")}
    if not {"导出对比 CSV", "下载配方报告 Markdown"}.issubset(comparison_controls):
        raise AssertionError("Recipe comparison downloads are missing")

    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    theme_source = (ROOT / "theme.py").read_text(encoding="utf-8")
    config_source = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    forbidden = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or "artifacts\\screenshots" in str(path) or path.suffix.lower() in {".png", ".pkl", ".db", ".pyc"}:
            continue
        try:
            forbidden_term = "\u8bba\u6587"
            if forbidden_term in path.read_text(encoding="utf-8"):
                forbidden.append(str(path.relative_to(ROOT)))
        except (UnicodeDecodeError, OSError):
            continue
    screenshot_paths = sorted(path for path in (ROOT / "artifacts" / "screenshots").rglob("*") if path.is_file())
    screenshot_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in screenshot_paths}
    try:
        status = subprocess.run(["git", "status", "--short", "--", "artifacts/screenshots"], cwd=ROOT, capture_output=True, text=True, check=False)
        screenshot_status = status.stdout.strip()
    except OSError:
        screenshot_status = "git unavailable"

    result = {
        "compensation_scan": {
            "elements_tested": len(scan_results),
            "all_elements_passed": all(item["all_totals_equal_100"] and not item["warnings"] for item in scan_results.values()),
            "details": scan_results,
        },
        "sample_prediction_MPa": float(direct["predicted_yield_strength_MPa"]),
        "comparison_matches_single_prediction": bool(comparison.iloc[0]["预测屈服强度 (MPa)"] == direct["predicted_yield_strength_MPa"]),
        "comparison_csv_openable": csv_openable,
        "comparison_markdown_openable": markdown_openable,
        "streamlit_page_count": len(rendered_pages),
        "streamlit_pages": rendered_pages,
        "zero_range_sensitivity_element": zero_range_element,
        "zero_range_sensitivity_rendered": True,
        "comparison_page_matches_single_prediction": ui_comparison_matches,
        "comparison_download_controls": sorted(comparison_controls),
        "extrapolation_annotation": bool("外推区，仅供参考" in app_source and "axvspan" in app_source),
        "no_generated_explanation_card": bool("归因说明" not in app_source and "机理解释" not in app_source and "生成式" not in app_source),
        "theme_constants_match_config": all(value in theme_source and value in config_source for value in ("#E8A317", "#7F77DD", "#1D9E75", "#E6E1D6", "#2C2C2A")),
        "screenshot_file_count": len(screenshot_paths),
        "screenshot_git_status": screenshot_status,
        "screenshot_hashes": screenshot_hashes,
        "forbidden_term_hits": forbidden,
    }
    output = ROOT / "artifacts" / "showcase_tests.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "compensation_scan"}, ensure_ascii=False, indent=2))
    print(json.dumps({"compensation_scan": result["compensation_scan"]["all_elements_passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
