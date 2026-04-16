"""
Car Damage Detection — Tahmin & Hasar Raporu Scripti
Kullanıcının gönderdiği araba görselini analiz eder.

Kullanım:
    python predict.py --image path/to/car.jpg
    python predict.py --image path/to/car.jpg --conf 0.3 --save
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """numpy tiplerini JSON-uyumlu Python tiplerine çevirir."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import cv2
import torch

# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
BEST_MODEL  = BASE_DIR / "runs" / "segment" / "car_damage_v2" / "weights" / "best.pt"
OUTPUT_DIR  = BASE_DIR / "predictions"
OUTPUT_DIR.mkdir(exist_ok=True)

# Hasar bölgesi renkleri (BGR)
DAMAGE_COLORS = [
    (86, 69, 233),    # kırmızı-mor
    (0, 165, 255),    # turuncu
    (0, 255, 127),    # yeşil
    (255, 0, 255),    # magenta
    (0, 255, 255),    # sarı
]
# ─────────────────────────────────────────────


def region_label(cx_norm, cy_norm):
    """
    Normalize merkez koordinatlarına göre hasar bölgesini belirle.
    Görsel 3×3 ızgaraya bölünür.
    """
    col = int(cx_norm * 3)  # 0=sol, 1=orta, 2=sağ
    row = int(cy_norm * 3)  # 0=üst, 1=orta, 2=alt
    col = min(col, 2)
    row = min(row, 2)

    col_labels = ["Sol", "Merkez", "Sağ"]
    row_labels = ["Üst",  "Orta",  "Alt"]
    return f"{row_labels[row]}-{col_labels[col]}"


def damage_severity(area_ratio):
    """Alan oranına göre hasar şiddetini belirle."""
    if area_ratio < 0.03:
        return "Hafif",  (0, 200, 100)   # yeşil
    elif area_ratio < 0.10:
        return "Orta",   (0, 165, 255)   # turuncu
    elif area_ratio < 0.25:
        return "Ağır",   (0, 80, 255)    # turuncu-kırmızı
    else:
        return "Çok Ağır", (0, 0, 220)   # kırmızı


def draw_results(img_bgr, detections):
    """
    Tespitleri görsel üzerine çizer:
    - Opaklıklı polygon overlay
    - Hasar bilgi kutusu
    - Numara etiketi
    """
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    output  = img_bgr.copy()

    for idx, det in enumerate(detections):
        color    = DAMAGE_COLORS[idx % len(DAMAGE_COLORS)]
        mask_pts = det["polygon_px"]

        if len(mask_pts) >= 3:
            pts = np.array(mask_pts, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2)

    # Opaklık karıştırma
    cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)

    # Her hasar için bilgi kutusu
    for idx, det in enumerate(detections):
        color    = DAMAGE_COLORS[idx % len(DAMAGE_COLORS)]
        cx, cy   = det["center_px"]
        severity, sev_color = damage_severity(det["area_ratio"])

        label = (f"#{idx+1} {det['region']} | {severity} "
                 f"| %{det['area_pct']:.1f} | {det['confidence']:.2f}")

        # Label arka planı
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx = max(0, cx - tw // 2)
        ly = max(th + 6, cy - 8)
        cv2.rectangle(output,
                      (lx - 4, ly - th - 4),
                      (lx + tw + 4, ly + 4),
                      (20, 20, 40), -1)
        cv2.putText(output, label,
                    (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Numara çemberi
        cv2.circle(output, (cx, cy), 16, color, -1)
        cv2.putText(output, str(idx + 1),
                    (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Alt bilgi bandı
    band_h = 36
    band   = np.zeros((band_h, w, 3), dtype=np.uint8)
    band[:] = (20, 20, 40)
    msg = (f"Toplam Hasar: {len(detections)} bolge  |  "
           f"Model: YOLOv8m-seg  |  Car Damage Detection")
    cv2.putText(band, msg, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
    output = np.vstack([output, band])

    return output


def predict(image_path: str, conf_threshold: float = 0.25, save: bool = True):
    """Görsel üzerinde tahmin yap ve hasar raporu üret."""
    from ultralytics import YOLO

    if not BEST_MODEL.exists():
        print(f"\n❌ Model bulunamadı: {BEST_MODEL}")
        print("   Önce: python train.py")
        sys.exit(1)

    if not Path(image_path).exists():
        print(f"❌ Görsel bulunamadı: {image_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = YOLO(str(BEST_MODEL))

    print(f"\n🔍 Tahmin çalıştırılıyor...")
    print(f"   Görsel  : {image_path}")
    print(f"   Conf    : {conf_threshold}")
    print(f"   Device  : {device}")

    results = model.predict(
        source = image_path,
        conf   = conf_threshold,
        device = device,
        imgsz  = 640,
        task   = "segment",
        verbose = False,
    )

    result  = results[0]
    img_bgr = cv2.imread(image_path)
    H, W    = img_bgr.shape[:2]

    detections = []

    if result.masks is not None:
        masks  = result.masks.xy         # polygon koordinatları (piksel)
        boxes  = result.boxes.xywhn      # normalize xywh
        confs  = result.boxes.conf.cpu().numpy()

        for i, (mask_pts, box, conf) in enumerate(zip(masks, boxes, confs)):
            mask_arr = np.array(mask_pts, dtype=np.float32)
            if len(mask_arr) == 0:
                continue

            # Normalize koordinatlar
            xs_norm = mask_arr[:, 0] / W
            ys_norm = mask_arr[:, 1] / H

            cx_norm = float(box[0])
            cy_norm = float(box[1])
            bw_norm = float(box[2])
            bh_norm = float(box[3])

            # Alan hesabı (Shoelace formülü)
            n = len(xs_norm)
            area_norm = 0.0
            for j in range(n):
                k = (j + 1) % n
                area_norm += xs_norm[j] * ys_norm[k]
                area_norm -= xs_norm[k] * ys_norm[j]
            area_norm = abs(area_norm) / 2

            detections.append({
                "id":          i + 1,
                "confidence":  float(conf),
                "center_norm": (cx_norm, cy_norm),
                "center_px":   (int(cx_norm * W), int(cy_norm * H)),
                "bbox_norm":   (cx_norm, cy_norm, bw_norm, bh_norm),
                "bbox_px":     (
                    int((cx_norm - bw_norm / 2) * W),
                    int((cy_norm - bh_norm / 2) * H),
                    int(bw_norm * W),
                    int(bh_norm * H),
                ),
                "area_ratio":  area_norm,
                "area_pct":    area_norm * 100,
                "region":      region_label(cx_norm, cy_norm),
                "polygon_px":  [(int(x), int(y)) for x, y in zip(
                    xs_norm * W, ys_norm * H
                )],
            })
    else:
        print("  ⚠️  Bu görselde hasar tespit edilemedi.")

    # ─── TERMİNAL RAPORU ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  🚗 HASAR TESPİT RAPORU")
    print("=" * 60)
    print(f"  Görsel          : {Path(image_path).name}")
    print(f"  Görsel Boyutu   : {W}×{H} px")
    print(f"  Tespit Sayısı   : {len(detections)}")
    print("-" * 60)

    if detections:
        for det in detections:
            severity, _ = damage_severity(det["area_ratio"])
            print(f"\n  [Hasar #{det['id']}]")
            print(f"    Bölge         : {det['region']}")
            print(f"    Şiddet        : {severity}")
            print(f"    Güven Skoru   : %{det['confidence']*100:.1f}")
            print(f"    Merkez (x,y)  : ({det['center_px'][0]}, {det['center_px'][1]}) px")
            bx, by, bw, bh = det["bbox_px"]
            print(f"    BBox (x,y,w,h): ({bx}, {by}, {bw}, {bh}) px")
            print(f"    Alan          : %{det['area_pct']:.2f} (görsel alanının)")
    else:
        print("  Hasar tespit edilemedi.")

    print("\n" + "=" * 60)

    # ─── JSON RAPORU ──────────────────────────────────────────
    report = {
        "timestamp":     datetime.now().isoformat(),
        "image_path":    image_path,
        "image_size":    {"width": W, "height": H},
        "model":         str(BEST_MODEL),
        "conf_threshold": conf_threshold,
        "damage_count":  len(detections),
        "damages": []
    }

    for det in detections:
        severity, _ = damage_severity(det["area_ratio"])
        report["damages"].append({
            "id":          det["id"],
            "region":      det["region"],
            "severity":    severity,
            "confidence":  round(det["confidence"], 4),
            "center_px":   list(det["center_px"]),
            "bbox_px":     list(det["bbox_px"]),
            "area_pct":    round(det["area_pct"], 3),
        })

    # ─── GÖRSEL KAYDET ────────────────────────────────────────
    if save:
        annotated = draw_results(img_bgr, detections)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem      = Path(image_path).stem
        out_img   = OUTPUT_DIR / f"{stem}_damage_{ts}.jpg"
        out_json  = OUTPUT_DIR / f"{stem}_report_{ts}.json"

        cv2.imwrite(str(out_img), annotated)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        print(f"  📸 Annotated görsel : {out_img}")
        print(f"  📄 JSON raporu      : {out_json}")
        print("=" * 60)

    return report, detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Damage Tespiti")
    parser.add_argument("--image", required=True, help="Analiz edilecek görsel yolu")
    parser.add_argument("--conf",  type=float, default=0.25, help="Confidence eşiği (0–1)")
    parser.add_argument("--save",  action="store_true", default=True,
                        help="Annotated görseli kaydet")
    args = parser.parse_args()

    predict(args.image, conf_threshold=args.conf, save=args.save)
