"""Shared visual constants for the Streamlit application and dynamic plots."""

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

PRIMARY = "#E8A317"
BACKGROUND = "#FFFDF7"
SECONDARY_BACKGROUND = "#FFF4D6"
PURPLE = "#7F77DD"
TEAL = "#1D9E75"
BORDER = "#E6E1D6"
TEXT = "#2C2C2A"
WARNING = "#B7791F"
MODEL_COLORS = {
    "LightGBM": PRIMARY,
    "RandomForest": PURPLE,
    "XGBoost": TEAL,
}
PLOT_GRID = BORDER
PLOT_DPI = 180


def configure_plot_font() -> str:
    candidates = (
        Path("C:/Windows/Fonts/Noto Sans SC (TrueType).otf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    for font_path in candidates:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except (OSError, RuntimeError):
                pass
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    preferred = ("Noto Sans SC", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei")
    selected = next((name for name in preferred if name in installed), "DejaVu Sans")
    mpl.rcParams["font.sans-serif"] = [selected, "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    return selected
