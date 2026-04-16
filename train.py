"""
Car Damage Detection — YOLOv8m-seg Eğitim Scripti
RTX 3060 (12GB VRAM) için optimize edilmiş ayarlar.

Kullanım:
    python train.py                    # tam eğitim (50 epoch)
    python train.py --epochs 5         # kısa test
    python train.py --eval-only        # sadece değerlendirme (best.pt gerekir)
    python train.py --resume           # kaldığı yerden devam
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATASET_DIR = BASE_DIR / "Car-Damage detection.v1i.yolov8"
DATA_YAML   = DATASET_DIR / "data.yaml"
FIXED_YAML  = BASE_DIR / "data_fixed.yaml"
RUN_DIR     = BASE_DIR / "runs" / "segment"
MODEL_NAME  = "yolov8m-seg.pt"
# ─────────────────────────────────────────────


def patch_data_yaml():
    """
    data.yaml içindeki relatif path'leri mutlak path'e çevir.
    Yeni dosya 'data_fixed.yaml' olarak kaydedilir.
    """
    with open(DATA_YAML, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["train"] = str(DATASET_DIR / "train"  / "images")
    cfg["val"]   = str(DATASET_DIR / "valid"  / "images")
    cfg["test"]  = str(DATASET_DIR / "test"   / "images")
    # YOLOv8-seg için label tipi
    cfg["task"]  = "segment"

    with open(FIXED_YAML, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"  ✓ data_fixed.yaml oluşturuldu: {FIXED_YAML}")
    print(f"    train : {cfg['train']}")
    print(f"    val   : {cfg['val']}")
    print(f"    test  : {cfg['test']}")
    return str(FIXED_YAML)


def check_gpu():
    """GPU durumunu kontrol et."""
    if torch.cuda.is_available():
        gpu_name    = torch.cuda.get_device_name(0)
        vram_gb     = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_ver    = torch.version.cuda
        print(f"\n🎮 GPU Bulundu!")
        print(f"   İsim     : {gpu_name}")
        print(f"   VRAM     : {vram_gb:.1f} GB")
        print(f"   CUDA     : {cuda_ver}")
        return "cuda"
    else:
        print("\n⚠️  GPU bulunamadı, CPU kullanılacak (eğitim çok yavaş olacak!)")
        return "cpu"


def train(epochs=50, batch=16, resume=False):
    """YOLOv8m-seg modelini eğit."""
    print("\n" + "=" * 60)
    print("  CAR DAMAGE — YOLOv8m-seg EĞİTİM")
    print("=" * 60)

    data_yaml = patch_data_yaml()
    device    = check_gpu()

    # Eğer resume ise mevcut best.pt'yi kullan
    if resume:
        last_weights = RUN_DIR / "car_damage_v2" / "weights" / "last.pt"
        if not last_weights.exists():
            print("❌ Devam edilecek model bulunamadı. Normal eğitim başlatılıyor...")
            resume = False
        else:
            MODEL_NAME_OR_PATH = str(last_weights)
            print(f"\n🔄 Eğitim devam ettiriliyor: {last_weights}")
    
    if not resume:
        MODEL_NAME_OR_PATH = MODEL_NAME

    model = YOLO(MODEL_NAME_OR_PATH)

    print(f"\n🚀 Eğitim Başlıyor...")
    print(f"   Model     : {MODEL_NAME_OR_PATH}")
    print(f"   Epochs    : {epochs}")
    print(f"   Batch     : {batch}")
    print(f"   Image sz  : 640")
    print(f"   Device    : {device}")
    print(f"   Workers   : 8")
    print("=" * 60 + "\n")

    results = model.train(
        data      = data_yaml,
        epochs    = epochs,
        imgsz     = 640,
        batch     = batch,
        device    = device,
        workers   = 4,            # Windows'ta 4 önerilir
        cache     = True,         # Görselleri RAM'e yükle → çok daha hızlı!
        optimizer = "AdamW",
        lr0       = 0.001,
        lrf       = 0.01,
        momentum  = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3,
        patience  = 15,           # early stopping
        project   = str(RUN_DIR),
        name      = "car_damage_v2",
        exist_ok  = True,
        save      = True,
        save_period = 10,         # her 10 epoch'ta checkpoint
        val       = True,
        plots     = True,
        task      = "segment",
        verbose   = True,
        amp       = True,         # Automatic Mixed Precision (RTX 3060 destekli)
        resume    = resume,
        # Augmentation
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 5.0,
        translate = 0.1,
        scale     = 0.4,
        flipud    = 0.0,
        fliplr    = 0.5,
        mosaic    = 0.8,
        mixup     = 0.0,
        copy_paste = 0.1,
    )

    # En iyi model yolu
    best_model = RUN_DIR / "car_damage_v2" / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print(f"  ✅ Eğitim Tamamlandı!")
    print(f"  📁 En iyi model: {best_model}")
    print("=" * 60)

    return results


def evaluate():
    """Test seti üzerinde modeli değerlendir."""
    best_model = RUN_DIR / "car_damage_v2" / "weights" / "best.pt"
    if not best_model.exists():
        print(f"❌ Model bulunamadı: {best_model}")
        print("   Önce python train.py ile eğitimi tamamlayın.")
        sys.exit(1)

    data_yaml = patch_data_yaml()
    print(f"\n📊 Test Seti Değerlendirme")
    print(f"   Model: {best_model}")

    model   = YOLO(str(best_model))
    device  = check_gpu()

    metrics = model.val(
        data   = data_yaml,
        split  = "test",
        device = device,
        imgsz  = 640,
        batch  = 8,
        task   = "segment",
        plots  = True,
        save_json = True,
    )

    print("\n" + "=" * 60)
    print("  📈 TEST METRİKLERİ")
    print("=" * 60)
    print(f"  mAP50 (bbox)  : {metrics.box.map50:.4f}")
    print(f"  mAP50 (seg)   : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95 (seg): {metrics.seg.map:.4f}")
    print(f"  Precision     : {metrics.seg.p[0]:.4f}")
    print(f"  Recall        : {metrics.seg.r[0]:.4f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Damage YOLOv8m-seg Eğitim")
    parser.add_argument("--epochs",    type=int,  default=50,    help="Epoch sayısı")
    parser.add_argument("--batch",     type=int,  default=8,     help="Batch boyutu (RTX3060 Laptop: 8)")
    parser.add_argument("--eval-only", action="store_true",      help="Sadece test değerlendirmesi")
    parser.add_argument("--resume",    action="store_true",      help="Kaldığı yerden devam et")
    args = parser.parse_args()

    if args.eval_only:
        evaluate()
    else:
        train(epochs=args.epochs, batch=args.batch, resume=args.resume)
