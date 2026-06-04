from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "cleaned_linefield_comparison"
GALLERY_DIR = OUT_DIR / "gallery"


def extract_xy_phi(mat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    m = loadmat(mat_path)
    segs = np.asarray(m["perpendicular_segments"], dtype=float)
    segs = segs[np.isfinite(segs).all(axis=1)]
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    xc = 0.5 * (x1 + x2)
    yc = 0.5 * (y1 + y2)
    phi = np.arctan2(-(y2 - y1), x2 - x1)
    X = np.column_stack([xc, yc])
    mask = np.isfinite(X).all(axis=1) & np.isfinite(phi)
    map_path = mat_path.with_name("arrow_maps.mat")
    if map_path.exists():
        maps = loadmat(map_path)
        h, w = np.asarray(maps["LBm"]).shape
        center = np.array([w / 2.0, h / 2.0], dtype=float)
        radius = 0.39 * float(min(h, w))
    else:
        center = np.median(X[mask], axis=0)
        radius = float(np.quantile(np.linalg.norm(X[mask] - center, axis=1), 0.9))
    return X[mask].astype(float), phi[mask].astype(float), center, radius


def central_roi_filter(X: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(X - center, axis=1) <= radius


def add_linefield(ax: plt.Axes, X: np.ndarray, phi: np.ndarray, color: str, center: np.ndarray, radius: float) -> None:
    if len(phi) == 0:
        return
    span = max(float(np.max(np.ptp(X, axis=0))), 1.0)
    seg_len = span / 42.0
    u = np.column_stack([np.cos(phi), np.sin(phi)]) * seg_len
    segments = np.stack([X - u, X + u], axis=1)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.85, alpha=0.78))
    ax.add_patch(Circle(center, radius, fill=False, edgecolor="#C44E52", linestyle="--", linewidth=1.0, alpha=0.8))
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_dataset_image(mat_path: Path, index: int) -> dict[str, str | int | float]:
    X, phi, center, radius = extract_xy_phi(mat_path)
    keep = central_roi_filter(X, center, radius)
    rel = str(mat_path.relative_to(DATA_DIR))
    img_name = f"{index:02d}_{rel.replace('/', '__').replace('.mat', '')}.png"
    img_path = GALLERY_DIR / img_name

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    add_linefield(axes[0], X, phi, "#8E8E8E", center, radius)
    axes[0].set_title(f"Raw ({len(phi)} segments)", fontsize=10)
    add_linefield(axes[1], X[keep], phi[keep], "#246B5A", center, radius)
    axes[1].set_title(f"Cleaned ({int(keep.sum())}, {keep.mean():.0%} retained)", fontsize=10)
    fig.suptitle(rel, fontsize=9)
    fig.tight_layout()
    fig.savefig(img_path, dpi=170)
    plt.close(fig)

    return {
        "dataset": rel,
        "image": f"gallery/{img_name}",
        "n_raw": int(len(phi)),
        "n_clean": int(keep.sum()),
        "retention": float(keep.mean()),
        "roi_center_x": float(center[0]),
        "roi_center_y": float(center[1]),
        "roi_radius": float(radius),
    }


def write_gallery(rows: list[dict[str, str | int | float]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<article>"
            f"<img src='{row['image']}' alt='{row['dataset']}'>"
            f"<h2>{row['dataset']}</h2>"
            f"<p>{row['n_clean']} / {row['n_raw']} retained "
            f"({float(row['retention']):.1%}); ROI center "
            f"({float(row['roi_center_x']):.1f}, {float(row['roi_center_y']):.1f}); "
            f"radius {float(row['roi_radius']):.1f}.</p>"
            "</article>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Cleaned Line-Field Gallery</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;max-width:1280px;margin:24px auto;line-height:1.45;color:#1f2933}"
        "header{margin-bottom:20px}h1{margin:0 0 6px}p{margin:6px 0}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}"
        "article{border:1px solid #d8dde3;padding:12px;background:#fff}"
        "img{width:100%;display:block;border:1px solid #edf0f2}h2{font-size:14px;margin:10px 0 4px;word-break:break-word}"
        "article p{font-size:13px;color:#52606d}"
        "</style></head><body>"
        "<header><h1>Cleaned Axial Line-Field Gallery</h1>"
        "<p>Each panel compares the original segment field with the central circular ROI field. The dashed red circle is the spatial cleaning boundary; segments outside it are discarded.</p></header>"
        "<section class='grid'>"
        + "\n".join(cards)
        + "</section></body></html>"
    )
    (OUT_DIR / "cleaned_gallery.html").write_text(html)
    pd.DataFrame(rows).to_csv(OUT_DIR / "cleaned_gallery_index.csv", index=False)


def main() -> None:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    mats = sorted(DATA_DIR.glob("**/arrow_segments.mat"))
    rows = [make_dataset_image(mat, i) for i, mat in enumerate(mats, 1)]
    write_gallery(rows)


if __name__ == "__main__":
    main()
