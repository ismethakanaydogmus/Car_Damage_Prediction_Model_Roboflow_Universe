"""
Car Damage Detection — Gradio Web Arayüzü
Kullanıcı araba fotoğrafı yükler, model analiz eder ve detaylı hasar raporu üretir.

Kullanım:
    python app.py
    Tarayıcıda: http://localhost:7860
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
import torch
from PIL import Image

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

# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
BEST_MODEL = BASE_DIR / "runs" / "segment" / "car_damage_v2" / "weights" / "best.pt"

DAMAGE_COLORS_RGB = [
    (233, 69, 86),    # kırmızı
    (255, 165, 0),    # turuncu
    (0, 200, 127),    # yeşil
    (200, 0, 255),    # magenta
    (0, 255, 220),    # cyan
    (255, 220, 0),    # sarı
]

_model = None
# ─────────────────────────────────────────────


def load_model():
    """Modeli bir kez yükle."""
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO
    if not BEST_MODEL.exists():
        return None
    _model = YOLO(str(BEST_MODEL))
    return _model


def region_label(cx_norm, cy_norm):
    col = min(int(cx_norm * 3), 2)
    row = min(int(cy_norm * 3), 2)
    col_labels = ["Sol", "Merkez", "Sağ"]
    row_labels = ["Üst", "Orta", "Alt"]
    return f"{row_labels[row]}-{col_labels[col]}"


def damage_severity(area_ratio):
    if area_ratio < 0.03:
        return "🟢 Hafif"
    elif area_ratio < 0.10:
        return "🟡 Orta"
    elif area_ratio < 0.25:
        return "🟠 Ağır"
    else:
        return "🔴 Çok Ağır"


def draw_annotated(img_bgr, detections):
    """Görsel üzerine hasar tespitlerini çiz."""
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    output  = img_bgr.copy()

    for idx, det in enumerate(detections):
        color_rgb = DAMAGE_COLORS_RGB[idx % len(DAMAGE_COLORS_RGB)]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        pts = np.array(det["polygon_px"], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color_bgr)
            cv2.polylines(output, [pts], isClosed=True, color=color_bgr, thickness=3)

    cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)

    for idx, det in enumerate(detections):
        color_rgb = DAMAGE_COLORS_RGB[idx % len(DAMAGE_COLORS_RGB)]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cx, cy = det["center_px"]

        # Numara çemberi
        cv2.circle(output, (cx, cy), 20, color_bgr, -1)
        cv2.circle(output, (cx, cy), 20, (255, 255, 255), 2)
        num_text = str(idx + 1)
        (tw, th), _ = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(output, num_text,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Bölge etiketi
        label = f"#{idx+1} {det['region']}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lx = max(0, cx - lw // 2)
        ly = max(lh + cy + 30, cy + 35)
        ly = min(ly, h - 5)
        cv2.rectangle(output,
                      (lx - 4, ly - lh - 4),
                      (lx + lw + 4, ly + 6),
                      (20, 20, 40), -1)
        cv2.putText(output, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # Alt şerit
    band_h = 40
    band   = np.zeros((band_h, w, 3), np.uint8)
    band[:] = (20, 20, 45)
    msg = (f"  Tespit: {len(detections)} hasar bolgesi  |  "
           f"YOLOv8m-seg  |  Car Damage AI")
    cv2.putText(band, msg, (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (180, 200, 255), 1, cv2.LINE_AA)
    output = np.vstack([output, band])
    return output


def analyze(image, conf_threshold):
    """Gradio callback: görsel → annotated görsel + tablo + JSON."""
    if image is None:
        return (None,
                "<p style='color:#e94560'>⚠️ Lütfen bir görsel yükleyin.</p>",
                "{}")

    model = load_model()
    if model is None:
        return (None,
                "<p style='color:#e94560'>❌ Model bulunamadı! "
                "Önce <code>python train.py</code> ile eğitimi tamamlayın.</p>",
                "{}")

    # PIL → NumPy BGR
    if isinstance(image, np.ndarray):
        img_rgb = image
    else:
        img_rgb = np.array(image)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W    = img_bgr.shape[:2]

    # Geçici dosyaya kaydet (ultralytics path ister)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        tmp_path = tmp.name

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    results  = model.predict(
        source  = tmp_path,
        conf    = conf_threshold,
        device  = device,
        imgsz   = 640,
        task    = "segment",
        verbose = False,
    )
    os.unlink(tmp_path)

    result     = results[0]
    detections = []

    if result.masks is not None:
        masks = result.masks.xy
        boxes = result.boxes.xywhn
        confs = result.boxes.conf.cpu().numpy()

        for i, (mask_pts, box, conf) in enumerate(zip(masks, boxes, confs)):
            mask_arr = np.array(mask_pts, dtype=np.float32)
            if len(mask_arr) == 0:
                continue

            xs_norm = mask_arr[:, 0] / W
            ys_norm = mask_arr[:, 1] / H
            cx_norm = float(box[0])
            cy_norm = float(box[1])
            bw_norm = float(box[2])
            bh_norm = float(box[3])

            # Shoelace alan formülü
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
                "center_px":   (int(cx_norm * W), int(cy_norm * H)),
                "bbox_px":     (
                    int((cx_norm - bw_norm / 2) * W),
                    int((cy_norm - bh_norm / 2) * H),
                    int(bw_norm * W),
                    int(bh_norm * H),
                ),
                "area_ratio":  area_norm,
                "area_pct":    area_norm * 100,
                "region":      region_label(cx_norm, cy_norm),
                "polygon_px":  [(int(x), int(y)) for x, y
                                in zip(mask_arr[:, 0], mask_arr[:, 1])],
            })

    # ─── Annotated görsel ──────────────────────────────────────
    annotated_bgr = draw_annotated(img_bgr, detections)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # ─── HTML Tablo Raporu ─────────────────────────────────────
    if not detections:
        html = """
        <div style='background:#1c1c2e;border-radius:12px;padding:24px;text-align:center;'>
          <p style='font-size:1.3rem;color:#aaa;'>✅ Bu görselde hasar tespit edilemedi.</p>
        </div>"""
    else:
        color_badges = [
            f"rgb({c[0]},{c[1]},{c[2]})" for c in DAMAGE_COLORS_RGB
        ]
        rows = ""
        for det in detections:
            severity  = damage_severity(det["area_ratio"])
            col_badge = color_badges[det["id"] - 1 % len(color_badges)]
            cx, cy    = det["center_px"]
            bx, by, bw, bh = det["bbox_px"]
            rows += f"""
            <tr>
              <td><span style='background:{col_badge};color:#fff;
                  border-radius:50%;padding:3px 9px;font-weight:bold;'>
                  {det["id"]}</span></td>
              <td><strong>{det["region"]}</strong></td>
              <td>{severity}</td>
              <td>{det["confidence"]*100:.1f}%</td>
              <td>({cx}, {cy}) px</td>
              <td>{bw}×{bh} px</td>
              <td>{det["area_pct"]:.2f}%</td>
            </tr>"""

        html = f"""
        <div style='background:#1a1a2e;border-radius:14px;padding:20px;'>
          <h3 style='color:#e94560;margin:0 0 14px 0;font-size:1.1rem;'>
            🚗 Hasar Tespit Raporu — {len(detections)} Hasar Bölgesi
          </h3>
          <table style='width:100%;border-collapse:collapse;font-size:0.9rem;color:#dde;'>
            <thead>
              <tr style='background:#0f3460;'>
                <th style='padding:8px 10px;text-align:left;border-radius:6px 0 0 0;'>#</th>
                <th style='padding:8px 10px;text-align:left;'>Bölge</th>
                <th style='padding:8px 10px;text-align:left;'>Şiddet</th>
                <th style='padding:8px 10px;text-align:left;'>Güven</th>
                <th style='padding:8px 10px;text-align:left;'>Merkez</th>
                <th style='padding:8px 10px;text-align:left;'>Boyut</th>
                <th style='padding:8px 10px;text-align:left;border-radius:0 6px 0 0;'>Alan</th>
              </tr>
            </thead>
            <tbody>
              {rows}
            </tbody>
          </table>
          <p style='color:#888;font-size:0.78rem;margin-top:10px;'>
            Görsel boyutu: {W}×{H} px &nbsp;|&nbsp;
            Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'} &nbsp;|&nbsp;
            Conf eşiği: {conf_threshold:.2f}
          </p>
        </div>"""

    # ─── JSON Raporu ──────────────────────────────────────────
    report_data = {
        "timestamp":     datetime.now().isoformat(),
        "image_size":    {"width": W, "height": H},
        "model":         str(BEST_MODEL),
        "conf_threshold": conf_threshold,
        "damage_count":  len(detections),
        "damages": [
            {
                "id":       d["id"],
                "region":   d["region"],
                "severity": damage_severity(d["area_ratio"]).split()[-1],
                "confidence": round(d["confidence"], 4),
                "center_px":  list(d["center_px"]),
                "bbox_px":    list(d["bbox_px"]),
                "area_pct":   round(d["area_pct"], 3),
            }
            for d in detections
        ]
    }
    json_str = json.dumps(report_data, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return annotated_rgb, html, json_str


# ─────────────────────────────────────────────
# Gradio Arayüzü
# ─────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #0d0d1a !important; }

#title-block {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 28px 32px 20px;
    margin-bottom: 20px;
    border: 1px solid #e94560;
}
#title-block h1 {
    color: #e94560;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
#title-block p {
    color: #9999bb;
    margin: 8px 0 0;
    font-size: 1rem;
}

.upload-box { border: 2px dashed #e94560 !important; border-radius: 14px !important; }
.upload-box:hover { border-color: #0f3460 !important; }

#conf-slider { accent-color: #e94560; }

#analyze-btn {
    background: linear-gradient(135deg, #e94560, #c22b47) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
}
#analyze-btn:hover {
    background: linear-gradient(135deg, #ff5577, #e94560) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(233,69,86,0.4) !important;
}

.label-text { color: #aabbd4 !important; font-weight: 600 !important; }

.output-image img { border-radius: 12px; }

.model-badge {
    display: inline-block;
    background: #0f3460;
    color: #7ecfff;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 8px;
}
"""

HEADER_HTML = """
<div id="title-block">
  <h1>🚗 Car Damage Detection AI</h1>
  <p>
    Araba fotoğrafını yükle → <strong>Hasar bölgelerini</strong> ve
    <strong>şiddetini</strong> anında analiz et.
    <br>
    <span class="model-badge">YOLOv8m-seg</span>
    <span class="model-badge">Instance Segmentation</span>
    <span class="model-badge">RTX 3060</span>
  </p>
</div>
"""

EXAMPLE_GUIDE = """
### 📋 Kullanım Kılavuzu
1. **Görsel Yükle** — Hasarlı araba fotoğrafını sol alana sürükle veya tıkla.  
2. **Güven Eşiği** — Düşük değer → daha fazla tespit (yanlış + da olabilir).  
   Yüksek değer → daha az ama emin tespitler.  
3. **Analiz Et** butonuna bas.  
4. **Sonuçlar** sağ tarafta görünür:
   - 📸 Annotated görsel (renkli polygon maskeler)  
   - 📊 Hasar raporu tablosu (bölge, şiddet, alan, konum)  
   - 🔵 JSON raporu (diğer sistemlere entegre etmek için)

### 🎯 Hasar Şiddeti
| İkon | Şiddet | Alan |
|------|--------|------|
| 🟢 | Hafif | < %3 |
| 🟡 | Orta | %3–10 |
| 🟠 | Ağır | %10–25 |
| 🔴 | Çok Ağır | > %25 |

### 📍 Bölge Haritası
Görsel 3×3 ızgaraya bölünür:
`Üst-Sol` | `Üst-Merkez` | `Üst-Sağ`  
`Orta-Sol` | `Orta-Merkez` | `Orta-Sağ`  
`Alt-Sol`  | `Alt-Merkez`  | `Alt-Sağ`
"""

def build_ui():
    with gr.Blocks(title="Car Damage Detection AI") as demo:

        gr.HTML(HEADER_HTML)

        with gr.Row():
            # ─── Sol Panel ─────────────────────────────────
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="🖼️ Araba Görseli Yükle",
                    type="numpy",
                    elem_classes=["upload-box"],
                    height=380,
                )
                conf_slider = gr.Slider(
                    minimum=0.10, maximum=0.90, value=0.25, step=0.05,
                    label="🎯 Güven Eşiği (Confidence Threshold)",
                    elem_id="conf-slider",
                )
                analyze_btn = gr.Button(
                    "🔍 Hasarı Analiz Et",
                    variant="primary",
                    elem_id="analyze-btn",
                )
                gr.Markdown(EXAMPLE_GUIDE)

            # ─── Sağ Panel ─────────────────────────────────
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="📸 Tespit Sonucu",
                    type="numpy",
                    elem_classes=["output-image"],
                    height=380,
                )
                html_report = gr.HTML(
                    value="<div style='background:#1a1a2e;border-radius:12px;"
                          "padding:20px;color:#666;text-align:center;'>"
                          "Analiz sonucu burada görünecek...</div>",
                    label="📊 Hasar Raporu",
                )
                json_output = gr.Code(
                    label="📄 JSON Raporu",
                    language="json",
                    lines=10,
                    value="{}",
                )

        analyze_btn.click(
            fn=analyze,
            inputs=[input_image, conf_slider],
            outputs=[output_image, html_report, json_output],
            api_name="analyze",
        )

        # Görsel değişince otomatik analiz (opsiyonel)
        # input_image.change(analyze, [input_image, conf_slider],
        #                    [output_image, html_report, json_output])

        gr.HTML("""
        <div style='text-align:center;padding:16px;color:#444;font-size:0.82rem;'>
          Car Damage Detection AI &nbsp;|&nbsp; YOLOv8m-seg &nbsp;|&nbsp;
          Roboflow Universe Dataset (11,685 görsel)
        </div>""")

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("  🚗 Car Damage Detection — Web Arayüzü")
    print("=" * 60)

    if not BEST_MODEL.exists():
        print(f"\n⚠️  Model bulunamadı: {BEST_MODEL}")
        print("   Lütfen önce eğitimi tamamlayın:")
        print("   python train.py")
        print("\n   Arayüz yine de başlatılıyor (model yükleme hatası gösterir).")

    device_str = "CUDA (GPU 🎮)" if torch.cuda.is_available() else "CPU"
    print(f"\n  Device: {device_str}")
    print(f"  Model : {BEST_MODEL}")
    print(f"  URL   : http://localhost:7860")
    print("=" * 60 + "\n")

    demo = build_ui()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        show_error  = True,
        css         = CSS,
        theme       = gr.themes.Base(
            primary_hue   = "red",
            secondary_hue = "blue",
            neutral_hue   = "slate",
            font          = ["Inter", "ui-sans-serif", "system-ui"],
        ),
    )
