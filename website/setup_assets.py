"""
Run this once from the repo root before pushing to GitHub:
    python website/setup_assets.py

Copies and web-optimises the curated charts into website/assets/charts/.
Requires Pillow: pip install Pillow
"""

import os
import shutil
from pathlib import Path

try:
    from PIL import Image
    USE_PIL = True
except ImportError:
    USE_PIL = False
    print("Pillow not found — copying PNGs as-is (run: pip install Pillow for web optimisation)")

ROOT  = Path(__file__).parent.parent
WEB   = Path(__file__).parent
DEST  = WEB / "assets" / "charts"
SRC   = ROOT / "08_outputs" / "charts"

MAX_WIDTH  = 1600   # px — sufficient for 1280px container @ 2x retina
JPEG_Q     = 88     # quality — good balance of size vs sharpness

CHART_MAP = {
    # (source subfolder, filename): destination subfolder
    ("01_eda_charts", "fig02_rating_decline_story.png"):    "eda",
    ("01_eda_charts", "fig03_cabin_performance.png"):       "eda",
    ("01_eda_charts", "fig04_score_drivers.png"):           "eda",
    ("01_eda_charts", "fig10_value_perception.png"):        "eda",
    ("01_eda_charts", "fig12_executive_summary.png"):       "eda",

    ("02_nlp_charts", "qatar_wordcloud_positive.png"):      "nlp",
    ("02_nlp_charts", "qatar_wordcloud_negative.png"):      "nlp",
    ("02_nlp_charts", "qatar_sentiment_trends.png"):        "nlp",
    ("02_nlp_charts", "fig_clusters_frequency.png"):        "nlp",

    ("03_ab_test_charts", "fig_ab_01_overview.png"):        "ab_test",
    ("03_ab_test_charts", "fig_ab_04_business_case.png"):   "ab_test",
    ("03_ab_test_charts", "fig_ab_05_executive_summary.png"): "ab_test",

    ("05_regression_charts", "fig_04_feature_importance_xgboost.png"): "regression",
    ("05_regression_charts", "fig_07_shap_summary.png"):    "regression",
    ("05_regression_charts", "fig_09_investment_framework.png"): "regression",
}


def process_image(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not USE_PIL:
        shutil.copy2(src_path, dst_path)
        print(f"  copied  {dst_path.relative_to(WEB)}")
        return

    img = Image.open(src_path).convert("RGB")
    if img.width > MAX_WIDTH:
        ratio  = MAX_WIDTH / img.width
        new_h  = int(img.height * ratio)
        img    = img.resize((MAX_WIDTH, new_h), Image.LANCZOS)

    # Save as JPEG for smaller size (PNGs from matplotlib are losslessly large)
    dst_jpg = dst_path.with_suffix(".jpg")
    img.save(dst_jpg, "JPEG", quality=JPEG_Q, optimize=True)
    size_kb = dst_jpg.stat().st_size // 1024
    print(f"  {dst_jpg.relative_to(WEB)}  ({size_kb} KB)")


def main():
    print(f"\nSource  : {SRC}")
    print(f"Dest    : {DEST}")
    print(f"Charts  : {len(CHART_MAP)}\n")

    ok, missing = 0, []
    for (subfolder, fname), dest_sub in CHART_MAP.items():
        src_path = SRC / subfolder / fname
        dst_name = Path(fname).with_suffix(".jpg") if USE_PIL else Path(fname)
        dst_path = DEST / dest_sub / dst_name

        if not src_path.exists():
            missing.append(str(src_path))
            print(f"  MISSING {src_path.relative_to(ROOT)}")
            continue

        process_image(src_path, dst_path)
        ok += 1

    print(f"\nDone: {ok}/{len(CHART_MAP)} charts processed.")
    if missing:
        print(f"Missing ({len(missing)}): run the relevant analysis scripts first.")


if __name__ == "__main__":
    main()
