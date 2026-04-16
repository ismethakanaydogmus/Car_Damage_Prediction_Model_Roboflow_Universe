"""
Car Damage Dataset - Analiz Scripti
Veri setini analiz eder, istatistikleri ve gorsellestirmeleri uretir.
"""

import sys
import io
# Windows terminal UTF-8 fix
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from PIL import Image
import cv2

# ─────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "Car-Damage detection.v1i.yolov8")
SPLITS   = {"train": "train", "validation": "valid", "test": "test"}
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────


def count_files(split_key):
    """Görsel ve etiket sayılarını döndür."""
    images_dir = os.path.join(BASE_DIR, SPLITS[split_key], "images")
    labels_dir = os.path.join(BASE_DIR, SPLITS[split_key], "labels")
    imgs   = glob.glob(os.path.join(images_dir, "*.*"))
    labels = glob.glob(os.path.join(labels_dir, "*.txt"))
    return len(imgs), len(labels)


def parse_labels(split_key):
    """
    Her label dosyasını okuyup annotation sayısı ve
    bounding-box boyutlarını (genişlik, yükseklik) döndür.
    YOLO-seg formatı: class_id x1 y1 x2 y2 ... xn yn
    """
    labels_dir = os.path.join(BASE_DIR, SPLITS[split_key], "labels")
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    ann_counts   = []  # her görseldeki annotation sayısı
    widths       = []  # normalize bbox genişlikleri
    heights      = []  # normalize bbox yükseklikleri
    areas        = []  # normalize bbox alanları

    for lf in label_files:
        with open(lf, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        ann_counts.append(len(lines))

        for line in lines:
            parts = list(map(float, line.split()))
            # class_id + polygon koordinatları
            coords = parts[1:]  # x1 y1 x2 y2 ...
            xs = coords[0::2]
            ys = coords[1::2]
            if len(xs) < 2:
                continue
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            widths.append(w)
            heights.append(h)
            areas.append(w * h)

    return ann_counts, widths, heights, areas


def size_category(w, h):
    """Hasar boyutunu kategorize et."""
    area = w * h
    if area < 0.05:
        return "Küçük"
    elif area < 0.15:
        return "Orta"
    else:
        return "Büyük"


def visualize_samples(split_key="train", n=6):
    """Rastgele n görsel + annotasyonunu çiz."""
    images_dir = os.path.join(BASE_DIR, SPLITS[split_key], "images")
    labels_dir = os.path.join(BASE_DIR, SPLITS[split_key], "labels")

    image_files = glob.glob(os.path.join(images_dir, "*.*"))
    sample_files = random.sample(image_files, min(n, len(image_files)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")
    axes = axes.flatten()

    colors = ["#e94560", "#0f3460", "#533483", "#16213e", "#e94560", "#533483"]

    for i, img_path in enumerate(sample_files):
        ax = axes[i]
        ax.set_facecolor("#16213e")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        ax.imshow(img)

        # Label dosyasını bul
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, basename + ".txt")

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]

            for j, line in enumerate(lines):
                parts = list(map(float, line.split()))
                coords = parts[1:]
                xs = [x * w for x in coords[0::2]]
                ys = [y * h for y in coords[1::2]]
                poly_pts = list(zip(xs, ys))

                color = colors[j % len(colors)]
                polygon = MplPolygon(poly_pts, closed=True,
                                     fill=True, alpha=0.3, color=color)
                ax.add_patch(polygon)
                polygon_border = MplPolygon(poly_pts, closed=True,
                                            fill=False, edgecolor=color,
                                            linewidth=2)
                ax.add_patch(polygon_border)

                # Merkez nokta
                cx = np.mean(xs)
                cy = np.mean(ys)
                bw = max(xs) - min(xs)
                bh = max(ys) - min(ys)
                cat = size_category(bw / w, bh / h)
                ax.text(cx, cy - 5, f"Hasar ({cat})",
                        color="white", fontsize=7,
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor=color, alpha=0.7))

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        ax.set_title(os.path.basename(img_path)[:30] + "...",
                     color="white", fontsize=8)

    plt.suptitle("Car Damage Dataset — Örnek Görsel Annotasyonları",
                 color="white", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "sample_annotations.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Örnek annotasyonlar kaydedildi: {out_path}")


def plot_statistics(all_stats):
    """İstatistik grafiklerini çiz."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#1a1a2e")
    plt.subplots_adjust(hspace=0.4, wspace=0.35)

    accent = "#e94560"
    blue   = "#0f3460"
    purple = "#533483"
    colors_bar = [accent, blue, purple]

    # 1. Bölünme Dağılımı (Bar)
    ax = axes[0, 0]
    ax.set_facecolor("#16213e")
    splits_names = list(all_stats.keys())
    img_counts   = [all_stats[s]["img_count"] for s in splits_names]
    bars = ax.bar(splits_names, img_counts, color=colors_bar, width=0.5,
                  edgecolor="white", linewidth=0.5)
    for bar, cnt in zip(bars, img_counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50, f"{cnt:,}",
                ha="center", color="white", fontsize=11, fontweight="bold")
    ax.set_title("Görsel Sayısı (Split)", color="white", fontsize=13, pad=10)
    ax.set_ylabel("Görsel Sayısı", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")
    ax.set_ylim(0, max(img_counts) * 1.15)

    # 2. Annotation / Görsel Dağılımı (Histogram)
    ax = axes[0, 1]
    ax.set_facecolor("#16213e")
    train_ann = all_stats["train"]["ann_counts"]
    ax.hist(train_ann, bins=range(0, max(train_ann) + 2),
            color=accent, edgecolor="#1a1a2e", linewidth=0.5, alpha=0.85)
    ax.axvline(np.mean(train_ann), color="white", linestyle="--",
               linewidth=1.5, label=f"Ort: {np.mean(train_ann):.2f}")
    ax.legend(facecolor="#16213e", labelcolor="white", fontsize=9)
    ax.set_title("Görsel Başına Hasar Sayısı (Train)", color="white", fontsize=13, pad=10)
    ax.set_xlabel("Hasar Adedi", color="white")
    ax.set_ylabel("Görsel Sayısı", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")

    # 3. BBox Genişlik Dağılımı
    ax = axes[0, 2]
    ax.set_facecolor("#16213e")
    train_w = all_stats["train"]["widths"]
    ax.hist(train_w, bins=50, color=blue, edgecolor="#1a1a2e",
            linewidth=0.3, alpha=0.85)
    ax.axvline(np.mean(train_w), color=accent, linestyle="--",
               linewidth=1.5, label=f"Ort: {np.mean(train_w):.3f}")
    ax.legend(facecolor="#16213e", labelcolor="white", fontsize=9)
    ax.set_title("Normalize Hasar Genişliği (Train)", color="white", fontsize=13, pad=10)
    ax.set_xlabel("Genişlik (0–1)", color="white")
    ax.set_ylabel("Frekans", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")

    # 4. BBox Yükseklik Dağılımı
    ax = axes[1, 0]
    ax.set_facecolor("#16213e")
    train_h = all_stats["train"]["heights"]
    ax.hist(train_h, bins=50, color=purple, edgecolor="#1a1a2e",
            linewidth=0.3, alpha=0.85)
    ax.axvline(np.mean(train_h), color=accent, linestyle="--",
               linewidth=1.5, label=f"Ort: {np.mean(train_h):.3f}")
    ax.legend(facecolor="#16213e", labelcolor="white", fontsize=9)
    ax.set_title("Normalize Hasar Yüksekliği (Train)", color="white", fontsize=13, pad=10)
    ax.set_xlabel("Yükseklik (0–1)", color="white")
    ax.set_ylabel("Frekans", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")

    # 5. Alan Scatter (Hasar Boyutu Dağılımı)
    ax = axes[1, 1]
    ax.set_facecolor("#16213e")
    tw = np.array(all_stats["train"]["widths"])
    th = np.array(all_stats["train"]["heights"])
    scatter = ax.scatter(tw, th, alpha=0.15, s=5, c=tw * th,
                         cmap="plasma", rasterized=True)
    plt.colorbar(scatter, ax=ax, label="Alan").ax.yaxis.label.set_color("white")
    ax.set_title("Hasar Boyutu Dağılımı (W×H)", color="white", fontsize=13, pad=10)
    ax.set_xlabel("Genişlik", color="white")
    ax.set_ylabel("Yükseklik", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")

    # 6. Boyut Kategorisi Pasta
    ax = axes[1, 2]
    ax.set_facecolor("#16213e")
    areas = all_stats["train"]["areas"]
    cats = {"Küçük (< 5%)": 0, "Orta (5–15%)": 0, "Büyük (> 15%)": 0}
    for a in areas:
        if a < 0.05:
            cats["Küçük (< 5%)"] += 1
        elif a < 0.15:
            cats["Orta (5–15%)"] += 1
        else:
            cats["Büyük (> 15%)"] += 1
    wedges, texts, autotexts = ax.pie(
        cats.values(), labels=cats.keys(),
        autopct="%1.1f%%", colors=[accent, blue, purple],
        textprops={"color": "white", "fontsize": 10},
        wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Hasar Boyut Kategorileri (Train)", color="white", fontsize=13, pad=10)

    plt.suptitle("Car Damage Dataset — İstatistik Analizi",
                 color="white", fontsize=18, fontweight="bold", y=1.02)

    out_path = os.path.join(OUTPUT_DIR, "dataset_statistics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ İstatistik grafikleri kaydedildi: {out_path}")


def main():
    print("=" * 60)
    print("  CAR DAMAGE DATASET — ANALİZ")
    print("=" * 60)

    all_stats = {}

    for split_name, split_dir in SPLITS.items():
        img_count, lbl_count = count_files(split_name)
        ann_counts, widths, heights, areas = parse_labels(split_name)

        all_stats[split_name] = {
            "img_count":  img_count,
            "lbl_count":  lbl_count,
            "ann_counts": ann_counts,
            "widths":     widths,
            "heights":    heights,
            "areas":      areas,
        }

        total_ann = sum(ann_counts)
        avg_ann   = np.mean(ann_counts) if ann_counts else 0
        avg_w     = np.mean(widths)  if widths  else 0
        avg_h     = np.mean(heights) if heights else 0
        avg_area  = np.mean(areas)   if areas   else 0

        print(f"\n📂 {split_name.upper()}")
        print(f"   Görsel sayısı     : {img_count:,}")
        print(f"   Label dosyası     : {lbl_count:,}")
        print(f"   Toplam annotation : {total_ann:,}")
        print(f"   Ort. hasar/görsel : {avg_ann:.2f}")
        print(f"   Ort. hasar w,h    : {avg_w:.3f}, {avg_h:.3f}")
        print(f"   Ort. hasar alanı  : %{avg_area*100:.2f}")

    print("\n📊 Grafikler oluşturuluyor...")
    visualize_samples("train", n=6)
    plot_statistics(all_stats)

    print("\n" + "=" * 60)
    print(f"  Çıktılar: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
