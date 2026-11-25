"""Shared visual constants for the Streamlit application and dynamic plots."""

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

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

# Bundled CJK font. Streamlit Community Cloud runs on Linux, where the
# Windows font paths below do not exist; without a bundled font, Matplotlib
# falls back to DejaVu Sans (no CJK glyphs) and every Chinese chart title or
# axis label renders as tofu boxes. Keep this file in the repo so charts
# render correctly on any platform.
BUNDLED_FONT = Path(__file__).resolve().parent / "assets" / "fonts" / "NotoSansSC-Regular.otf"


def configure_plot_font() -> str:
    if BUNDLED_FONT.exists():
        try:
            font_manager.fontManager.addfont(str(BUNDLED_FONT))
        except (OSError, RuntimeError):
            pass
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


def cjk_fontproperties() -> FontProperties | None:
    """FontProperties bound directly to the bundled font file.

    Runtime-drawn charts (single-prediction sensitivity curve) cannot use
    pre-rendered PNGs, so they still need a working CJK font at runtime.
    The rcParams registration in configure_plot_font() proved unreliable
    on Streamlit Cloud: on 2026-11-25 the About-page histogram and the
    sensitivity curve still rendered as tofu boxes even though the font
    file was present in the repo and the same code rendered correctly on
    Linux with Matplotlib 3.11.2. Pointing FontProperties at the file
    bypasses the font-manager registry lookup entirely. Returns None when
    the file is absent so callers fall back to the default font.
    """
    if BUNDLED_FONT.exists():
        try:
            return FontProperties(fname=str(BUNDLED_FONT))
        except (OSError, RuntimeError, ValueError):
            return None
    return None


CJK_FONT = cjk_fontproperties()
