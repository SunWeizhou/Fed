#!/usr/bin/env python3
"""Generate thesis figures for the finalized FedViM thesis storyline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image, ImageOps


DISPLAY_MODEL_NAMES = {
    "resnet101": "ResNet101",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "mobilenetv3_large": "MobileNetV3-Large",
    "densenet169": "DenseNet169",
    "resnet50": "ResNet50",
}

DISPLAY_MODEL_NAMES_MULTILINE = {
    "resnet101": "ResNet101",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "mobilenetv3_large": "MobileNetV3-\nLarge",
    "densenet169": "DenseNet169",
    "resnet50": "ResNet50",
}

COLORS = {
    "FedViM": "#214C63",
    "ACT-FedViM": "#D45D47",
    "MSP": "#7EA66A",
    "Energy": "#C9A33A",
    "FedLN": "#7EA66A",
    "FOSTER": "#C9A33A",
}

DATASET_HEADER_COLORS = {
    "ID": "#214C63",
    "Near-OOD": "#C9A33A",
    "Far-OOD": "#7EA66A",
}

DATASET_PANEL_FILLS = {
    "ID": "#ECEAF0",
    "Near-OOD": "#F2EEDF",
    "Far-OOD": "#E8EFE7",
}

DATASET_PANEL_SPECS = [
    {
        "title": "ID",
        "count_label": "54 types",
        "rows": [
            [
                {"split": "D_ID_train", "class_name": "011_Acartia sp.A", "label": "Acartia sp", "image_index": 18},
                {"split": "D_ID_train", "class_name": "071_Creseis acicula", "label": "Creseis acicula", "image_index": 24},
                {"split": "D_ID_train", "class_name": "014_Calanopia sp", "label": "Calanopia sp", "image_index": 31},
            ],
            [
                {"split": "D_ID_train", "class_name": "036_Amphipoda_Type A", "label": "Amphipoda", "image_index": 12},
                {"split": "D_ID_train", "class_name": "060_Cumacea_Type A", "label": "Cumacea", "image_index": 19},
                {"split": "D_ID_train", "class_name": "042_Cymodoce sp", "label": "Cymodoce", "image_index": 6},
            ],
            [
                {"split": "D_ID_train", "class_name": "068_Jellyfish", "label": "Jellyfish", "image_index": 14},
                {"split": "D_ID_train", "class_name": "044_Macrura larvae", "label": "Macrura larva", "image_index": 42},
            ],
        ],
    },
    {
        "title": "Near-OOD",
        "count_label": "26 types",
        "rows": [
            [
                {"split": "D_Near_test", "class_name": "059_Ostracoda", "label": "Ostracoda", "image_index": 9},
                {"split": "D_Near_test", "class_name": "002_Polychaeta larva", "label": "Polychaeta larva", "image_index": 7},
            ],
            [
                {"split": "D_Near_test", "class_name": "067_Hydroid", "label": "Hydroid", "image_index": 3},
                {"split": "D_Near_test", "class_name": "028_Harpacticoid", "label": "Harpacticoid", "image_index": 4},
            ],
            [
                {"split": "D_Near_test", "class_name": "018_Calanoid Nauplii", "label": "Calanoid Nauplii", "image_index": 5},
                {"split": "D_Near_test", "class_name": "021_Calanoid_Type C", "label": "Calanoid", "image_index": 5},
            ],
        ],
    },
    {
        "title": "Far-OOD",
        "count_label": "12 types",
        "grouped_rows": [
            {
                "label": "Particles",
                "items": [
                    {"split": "D_Far_test", "class_name": "085_Particle_bluish", "image_index": 8},
                    {"split": "D_Far_test", "class_name": "087_Particle_translucent flocs", "image_index": 11},
                ],
            },
            {
                "label": "Crustacean limb",
                "items": [
                    {"split": "D_Far_test", "class_name": "078_Crustacean limb_Type A", "image_index": 4},
                    {"split": "D_Far_test", "class_name": "079_Crustacean limb_Type B", "image_index": 7},
                ],
            },
            {
                "label": "Bubbles",
                "items": [
                    {"split": "D_Far_test", "class_name": "090_Bubbles", "image_index": 45},
                    {"split": "D_Far_test", "class_name": "090_Bubbles", "image_index": 127},
                ],
            },
        ],
    },
]

CASE_NOTES = {
    "mobilenetv3_large": "Large compression with nearly unchanged performance",
    "resnet101": "Balanced compression-performance tradeoff",
    "densenet169": "Representative fixed-k mismatch correction case",
}

DATASET_TILE_SIZE_SCALE = 1.875

EDGE_COLOR = "#23313C"
GRID_COLOR = "#D7DEE5"
AXIS_FACE = "#F7F7F3"
TEXT_COLOR = "#1F2933"
TEXT_SUBTLE = "#5B6770"
CONNECTOR_COLOR = "#BFC9D4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate finalized thesis figures from manifest outputs.")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="paper_tools/manifest_final_mainline_summary.json",
        help="Mainline summary JSON generated from finalized manifests.",
    )
    parser.add_argument(
        "--comparison-json",
        type=str,
        default="paper_tools/manifest_final_mainline_full_comparison.json",
        help="Mainline per-model comparison JSON.",
    )
    parser.add_argument(
        "--selected-json",
        type=str,
        default="paper_tools/manifest_final_mainline_selected_models.json",
        help="Representative-model JSON.",
    )
    parser.add_argument(
        "--baselines-json",
        type=str,
        default="paper_tools/manifest_final_v1_main_vs_baselines.json",
        help="Mainline vs federated baselines JSON.",
    )
    parser.add_argument(
        "--pooled-json",
        type=str,
        default="paper_tools/manifest_final_v1_pooled_consistency.json",
        help="Federated vs pooled consistency JSON.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="Plankton_OOD_Dataset",
        help="Dataset root used to pick representative image examples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_tools/figures",
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, facecolor: str = "white") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=350, bbox_inches="tight", facecolor=facecolor)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor=facecolor)
    plt.close(fig)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": AXIS_FACE,
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.linewidth": 0.9,
            "axes.titleweight": "semibold",
            "font.family": ["DejaVu Serif", "DejaVu Sans"],
            "font.size": 11,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linestyle": (0, (3, 3)),
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def display_model_name(name: str, multiline: bool = False) -> str:
    if multiline:
        return DISPLAY_MODEL_NAMES_MULTILINE.get(name, name)
    return DISPLAY_MODEL_NAMES.get(name, name)


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.tick_params(length=0)


def add_bar_labels(ax: plt.Axes, bars, dy: float = 0.22) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_SUBTLE,
        )


def first_image_in_dir(path: Path) -> Path:
    images = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise FileNotFoundError(f"No images found in {path}")
    return images[0]


def nth_image_in_dir(path: Path, index: int = 0) -> Path:
    images = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise FileNotFoundError(f"No images found in {path}")
    bounded_index = max(0, min(index, len(images) - 1))
    return images[bounded_index]


def load_square_example(image_path: Path, size: int = 340, zoom_factor: float = 1.0) -> np.ndarray:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image)
        foreground_mask = array.max(axis=2) > 18

        if foreground_mask.any():
            ys, xs = np.where(foreground_mask)
            x_min, x_max = xs.min(), xs.max() + 1
            y_min, y_max = ys.min(), ys.max() + 1
            pad_x = max(6, int((x_max - x_min) * 0.18))
            pad_y = max(6, int((y_max - y_min) * 0.18))
            image = image.crop(
                (
                    max(0, x_min - pad_x),
                    max(0, y_min - pad_y),
                    min(image.width, x_max + pad_x),
                    min(image.height, y_max + pad_y),
                )
            )

        if zoom_factor > 1.0:
            zoom_w = max(8, int(round(image.width / zoom_factor)))
            zoom_h = max(8, int(round(image.height / zoom_factor)))
            left = max(0, (image.width - zoom_w) // 2)
            top = max(0, (image.height - zoom_h) // 2)
            right = min(image.width, left + zoom_w)
            bottom = min(image.height, top + zoom_h)
            image = image.crop((left, top, right, bottom))

        image.thumbnail((size - 20, size - 20), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (8, 8, 10))
        x_offset = (size - image.width) // 2
        y_offset = (size - image.height) // 2
        canvas.paste(image, (x_offset, y_offset))
    return np.asarray(canvas)


def add_example_tile(
    root_ax: plt.Axes,
    image: np.ndarray,
    rect: tuple[float, float, float, float],
    border_color: str,
    border_width: float = 2.8,
) -> None:
    tile_ax = root_ax.inset_axes(rect)
    tile_ax.imshow(image)
    tile_ax.set_xticks([])
    tile_ax.set_yticks([])
    tile_ax.set_facecolor("#050608")
    for spine in tile_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(border_width)
        spine.set_color(border_color)


def add_tile_label(root_ax: plt.Axes, x: float, y: float, text: str, fontsize: float = 14.0) -> None:
    root_ax.text(
        x,
        y,
        text,
        ha="center",
        va="top",
        fontsize=fontsize,
        color=TEXT_COLOR,
    )


def draw_dataset_panel(
    root_ax: plt.Axes,
    panel_box: tuple[float, float, float, float],
    panel_spec: dict,
    data_root: Path,
) -> None:
    x0, y0, width, height = panel_box
    title = panel_spec["title"]
    border_color = DATASET_HEADER_COLORS[title]
    fill_color = DATASET_PANEL_FILLS[title]

    panel = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        linewidth=2.2,
        edgecolor=border_color,
        facecolor=fill_color,
        linestyle=(0, (9, 5)),
        transform=root_ax.transAxes,
        zorder=0,
    )
    root_ax.add_patch(panel)

    root_ax.text(
        x0 + width / 2,
        y0 + height - 0.055,
        title,
        transform=root_ax.transAxes,
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=border_color,
    )
    root_ax.text(
        x0 + width / 2,
        y0 + 0.022,
        panel_spec["count_label"],
        transform=root_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=18,
        color=TEXT_COLOR,
    )

    if "rows" in panel_spec:
        rows = panel_spec["rows"]
        max_cols = max(len(row) for row in rows)
        base_tile_size = min(width * 0.185, height * 0.22) if title == "ID" else min(width * 0.31, height * 0.21)
        tile_size = base_tile_size * DATASET_TILE_SIZE_SCALE
        row_y = {
            3: [y0 + height * 0.70, y0 + height * 0.44, y0 + height * 0.14],
            2: [y0 + height * 0.64, y0 + height * 0.28],
        }[len(rows)]

        for row_idx, row in enumerate(rows):
            if title == "ID" and len(row) == 2:
                xs = [x0 + width * 0.29, x0 + width * 0.66]
            elif max_cols == 3:
                xs = [x0 + width * 0.22, x0 + width * 0.50, x0 + width * 0.78][: len(row)]
            else:
                xs = [x0 + width * 0.30, x0 + width * 0.70][: len(row)]

            for x_center, item in zip(xs, row):
                image_path = nth_image_in_dir(
                    data_root / item["split"] / item["class_name"],
                    item.get("image_index", 0),
                )
                image = load_square_example(image_path)
                rect = (
                    x_center - tile_size / 2,
                    row_y[row_idx],
                    tile_size,
                    tile_size,
                )
                add_example_tile(root_ax, image, rect, border_color)
                add_tile_label(
                    root_ax,
                    x_center,
                    row_y[row_idx] - 0.012,
                    item["label"],
                    fontsize=13.5 if title == "ID" else 13.0,
                )
        return

    tile_size = min(width * 0.34, height * 0.20) * DATASET_TILE_SIZE_SCALE
    row_y = [y0 + height * 0.69, y0 + height * 0.43, y0 + height * 0.16]
    x_centers = [x0 + width * 0.30, x0 + width * 0.72]
    for y_base, row in zip(row_y, panel_spec["grouped_rows"]):
        for x_center, item in zip(x_centers, row["items"]):
            image_path = nth_image_in_dir(
                data_root / item["split"] / item["class_name"],
                item.get("image_index", 0),
            )
            image = load_square_example(image_path)
            rect = (
                x_center - tile_size / 2,
                y_base,
                tile_size,
                tile_size,
            )
            add_example_tile(root_ax, image, rect, border_color)
        add_tile_label(root_ax, x0 + width / 2, y_base - 0.03, row["label"], fontsize=15.5)


def plot_dataset_examples(data_root: Path, output_dir: Path) -> None:
    # Match the reference composition ratio and panel geometry.
    fig = plt.figure(figsize=(16.1, 7.0), facecolor="white")
    root_ax = fig.add_axes([0, 0, 1, 1], facecolor="white")
    root_ax.set_xlim(0, 1)
    root_ax.set_ylim(0, 1)
    root_ax.set_axis_off()

    panel_boxes = {
        "ID": (0.024, 0.030, 0.408, 0.935),
        "Near-OOD": (0.444, 0.030, 0.274, 0.935),
        "Far-OOD": (0.732, 0.030, 0.242, 0.935),
    }

    for panel_spec in DATASET_PANEL_SPECS:
        draw_dataset_panel(root_ax, panel_boxes[panel_spec["title"]], panel_spec, data_root)

    save_figure(fig, output_dir, "figure_1_dataset_examples")


def plot_main_vs_baselines(rows: list[dict], output_dir: Path) -> None:
    models = [display_model_name(row["model_name"], multiline=True) for row in rows]
    x = np.arange(len(models))
    width = 0.18

    series = {
        "Near-OOD AUROC": {
            "FedViM": [row["fedvim_near_auroc"] * 100 for row in rows],
            "ACT-FedViM": [row["act_near_auroc"] * 100 for row in rows],
            "FedLN": [row["fedln_near_auroc"] * 100 for row in rows],
            "FOSTER": [row["foster_near_auroc"] * 100 for row in rows],
        },
        "Far-OOD AUROC": {
            "FedViM": [row["fedvim_far_auroc"] * 100 for row in rows],
            "ACT-FedViM": [row["act_far_auroc"] * 100 for row in rows],
            "FedLN": [row["fedln_far_auroc"] * 100 for row in rows],
            "FOSTER": [row["foster_far_auroc"] * 100 for row in rows],
        },
    }

    fig, axes = plt.subplots(2, 1, figsize=(12.2, 8.0), sharex=True, constrained_layout=True)
    for ax, (title, panel) in zip(axes, series.items()):
        for idx, (label, values) in enumerate(panel.items()):
            offset = (idx - 1.5) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=COLORS[label],
                edgecolor=EDGE_COLOR,
                linewidth=0.55,
                zorder=3,
                label=label,
            )
        style_axis(ax, grid_axis="y")
        ax.set_ylim(70, 99)
        ax.set_ylabel("AUROC (%)")
        ax.set_title(title, loc="left", pad=10, fontsize=13)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(models)
    legend_handles = [
        Patch(facecolor=COLORS["FedViM"], edgecolor=EDGE_COLOR, label="FedViM"),
        Patch(facecolor=COLORS["ACT-FedViM"], edgecolor=EDGE_COLOR, label="ACT-FedViM"),
        Patch(facecolor=COLORS["FedLN"], edgecolor=EDGE_COLOR, label="FedLN"),
        Patch(facecolor=COLORS["FOSTER"], edgecolor=EDGE_COLOR, label="FOSTER"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.8,
        handletextpad=0.6,
    )
    save_figure(fig, output_dir, "figure_2_main_vs_baselines")


def plot_pooled_consistency(rows: list[dict], output_dir: Path) -> None:
    models = [display_model_name(row["model_name"]) for row in rows]
    y = np.arange(len(rows))
    all_deltas = []
    for row in rows:
        all_deltas.extend(
            [
                row["delta_vim_near"] * 100,
                row["delta_vim_far"] * 100,
                row["delta_act_near"] * 100,
                row["delta_act_far"] * 100,
            ]
        )
    max_abs_delta = max(abs(v) for v in all_deltas)
    xlim = max(0.6, max_abs_delta * 1.15)

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.2), constrained_layout=True)
    panels = [
        (
            axes[0],
            "FedViM vs Pooled-ViM",
            [row["delta_vim_near"] * 100 for row in rows],
            [row["delta_vim_far"] * 100 for row in rows],
            COLORS["FedViM"],
        ),
        (
            axes[1],
            "ACT-FedViM vs Pooled-ACT-ViM",
            [row["delta_act_near"] * 100 for row in rows],
            [row["delta_act_far"] * 100 for row in rows],
            COLORS["ACT-FedViM"],
        ),
    ]

    for ax, title, near_vals, far_vals, color in panels:
        ax.axvline(0, color=CONNECTOR_COLOR, linewidth=1.6, zorder=1)
        ax.hlines(y, 0, near_vals, color=color, linewidth=1.5, alpha=0.28, zorder=2)
        ax.scatter(near_vals, y, s=64, color=color, edgecolor="white", linewidth=0.7, zorder=4, label="Near-OOD")
        ax.scatter(
            far_vals,
            y,
            s=64,
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            marker="s",
            zorder=4,
            label="Far-OOD",
        )
        style_axis(ax, grid_axis="x")
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.invert_yaxis()
        ax.set_xlim(-xlim, xlim)
        ax.set_title(title, loc="left", pad=10, fontsize=13)
        ax.set_xlabel("Pooled - Federated (percentage points)")

    legend_handles = [
        Line2D([0], [0], marker="o", markersize=7, color="none", markerfacecolor=TEXT_COLOR, label="Near-OOD"),
        Line2D(
            [0],
            [0],
            marker="s",
            markersize=7,
            color="none",
            markerfacecolor="white",
            markeredgecolor=TEXT_COLOR,
            label="Far-OOD",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.8,
        handletextpad=0.6,
    )
    save_figure(fig, output_dir, "figure_3_pooled_consistency")


def plot_subspace_compression(rows: list[dict], output_dir: Path) -> None:
    models = [display_model_name(row["model_name"]) for row in rows]
    fixed_k = [row["fedvim_fixed_k"] for row in rows]
    act_k = [row["act_k"] for row in rows]
    compression = [row["act_compression_rate"] * 100 for row in rows]
    y = np.arange(len(rows))
    max_fixed = max(fixed_k)
    right_label_x = max_fixed + 38

    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.86, bottom=0.14)
    ax.set_facecolor("#F6F7F4")

    for idx, (fixed_value, act_value, rate) in enumerate(zip(fixed_k, act_k, compression)):
        ax.hlines(idx, act_value, fixed_value, color=CONNECTOR_COLOR, linewidth=5.5, zorder=1, capstyle="round")
        ax.scatter(
            fixed_value,
            idx,
            s=150,
            facecolor="white",
            edgecolor=COLORS["FedViM"],
            linewidth=2.1,
            zorder=3,
        )
        ax.scatter(
            act_value,
            idx,
            s=150,
            facecolor=COLORS["ACT-FedViM"],
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            right_label_x,
            idx,
            f"-{rate:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            color=TEXT_SUBTLE,
        )

    style_axis(ax, grid_axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlim(0, max_fixed + 170)
    ax.set_xlabel("Subspace dimension k")
    ax.set_title("Subspace compression under ACT selection", loc="left", pad=10, fontsize=13)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=COLORS["FedViM"],
            markeredgewidth=2,
            linestyle="None",
            label="FedViM fixed-k",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=8,
            markerfacecolor=COLORS["ACT-FedViM"],
            markeredgecolor="white",
            linestyle="None",
            label="ACT-selected k",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.11),
        ncol=2,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    save_figure(fig, output_dir, "figure_4_subspace_compression")


def plot_selected_models(selected_rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.24, wspace=0.03)
    categories = ["Near-OOD", "Far-OOD"]
    x = np.arange(len(categories))
    width = 0.28

    for ax, row in zip(axes, selected_rows):
        fed_values = [row["fedvim_near_auroc"] * 100, row["fedvim_far_auroc"] * 100]
        act_values = [row["act_near_auroc"] * 100, row["act_far_auroc"] * 100]

        fed_bars = ax.bar(
            x - width / 2,
            fed_values,
            width=width,
            color=COLORS["FedViM"],
            edgecolor=EDGE_COLOR,
            linewidth=0.55,
            zorder=3,
            label="FedViM",
        )
        act_bars = ax.bar(
            x + width / 2,
            act_values,
            width=width,
            color=COLORS["ACT-FedViM"],
            edgecolor=EDGE_COLOR,
            linewidth=0.55,
            zorder=3,
            label="ACT-FedViM",
        )

        style_axis(ax, grid_axis="y")
        ax.set_ylim(75, 99)
        ax.set_xticks(x)
        ax.set_xticklabels(["Near", "Far"])
        ax.set_title(display_model_name(row["model_name"]), fontsize=13, pad=28)
        ax.text(
            0.5,
            1.005,
            CASE_NOTES[row["model_name"]],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.3,
            color=TEXT_SUBTLE,
        )
        ax.text(
            0.5,
            -0.18,
            f"k {row['fedvim_fixed_k']} -> {row['act_k']}  |  -{row['act_compression_rate'] * 100:.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.0,
            color=TEXT_SUBTLE,
        )
        add_bar_labels(ax, fed_bars)
        add_bar_labels(ax, act_bars)

    axes[0].set_ylabel("AUROC (%)")
    legend_handles = [
        Patch(facecolor=COLORS["FedViM"], edgecolor=EDGE_COLOR, label="FedViM"),
        Patch(facecolor=COLORS["ACT-FedViM"], edgecolor=EDGE_COLOR, label="ACT-FedViM"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.97),
        columnspacing=1.8,
        handletextpad=0.6,
    )
    save_figure(fig, output_dir, "figure_5_selected_models")


def write_caption_notes(summary: dict, output_dir: Path) -> None:
    content = f"""# Figure Captions

## Figure 1

Representative image examples from the ID, Near-OOD, and Far-OOD splits used in the thesis experiments.

## Figure 2

Five-model Near-OOD and Far-OOD AUROC comparison for `FedViM`, `ACT-FedViM`, `FedLN`, and `FOSTER`.

## Figure 3

Per-model consistency between federated and pooled statistics. The horizontal axis reports `Pooled - Federated` in percentage points.

## Figure 4

Comparison between the fixed-k FedViM subspace dimension and the ACT-selected subspace dimension. The average compression rate is {summary['act_vs_fedvim']['avg_compression_rate'] * 100:.2f}% across the five CNN backbones.

## Figure 5

Representative thesis models covering three complementary cases: lightweight deployment (`MobileNetV3-Large`), balanced compression-performance tradeoff (`ResNet101`), and fixed-k mismatch correction (`DenseNet169`).
"""
    (output_dir / "figure_captions.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = load_json(args.summary_json)
    comparison_rows = load_json(args.comparison_json)
    selected_rows = load_json(args.selected_json)
    baselines_rows = load_json(args.baselines_json)["rows"]
    pooled_rows = load_json(args.pooled_json)["rows"]
    output_dir = Path(args.output_dir)

    configure_matplotlib()
    plot_dataset_examples(Path(args.data_root), output_dir)
    plot_main_vs_baselines(baselines_rows, output_dir)
    plot_pooled_consistency(pooled_rows, output_dir)
    plot_subspace_compression(comparison_rows, output_dir)
    plot_selected_models(selected_rows, output_dir)
    write_caption_notes(summary, output_dir)

    print(f"Generated figures in: {output_dir}")


if __name__ == "__main__":
    main()
