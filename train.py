"""
Car Damage Detection — YOLOv8m-seg Eğitim Scripti
RTX 3060 (12GB VRAM) için optimize edilmiş ayarlar.

Kullanım:
    python merge_datasets.py            # önce veri setlerini birleştir (bir kere)
    python train.py                     # tam eğitim (50 epoch, best.pt üzerinden)
    python train.py --epochs 30         # daha kısa eğitim
    python train.py --eval-only         # sadece değerlendirme (best.pt gerekir)
    python train.py --resume            # kaldığı yerden devam
    python train.py --scratch           # backbone'u da sıfırla (önerilmez)
"""

import sys
import argparse
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MERGED_DIR   = BASE_DIR / "merged_dataset"
DATA_YAML    = MERGED_DIR / "data.yaml"
RUN_DIR      = BASE_DIR / "runs" / "segment"
PRETRAINED   = RUN_DIR / "car_damage_v2" / "weights" / "best.pt"
SCRATCH_MODEL = "yolov8m-seg.pt"
RUN_NAME     = "car_damage_v3"
# ─────────────────────────────────────────────


def check_merged_dataset():
    """Birleştirilmiş veri setinin hazır olup olmadığını kontrol et."""
    if not MERGED_DIR.exists() or not DATA_YAML.exists():
        print("❌ merged_dataset/ bulunamadı.")
        print("   Önce şu komutu çalıştırın: python merge_datasets.py")
        sys.exit(1)

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"\n  Veri seti : {MERGED_DIR.name}")
    print(f"  Sınıf sayısı: {cfg['nc']}")
    print(f"  Sınıflar  : {', '.join(cfg['names'])}")
    return str(DATA_YAML)


def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU      : {gpu_name}")
        print(f"  VRAM     : {vram_gb:.1f} GB")
        print(f"  CUDA     : {torch.version.cuda}")
        return "cuda"
    else:
        print("\n  ⚠️  GPU bulunamadı — CPU kullanılacak (eğitim çok yavaş olacak!)")
        return "cpu"


def train(epochs=50, batch=8, resume=False, scratch=False):
    print("\n" + "=" * 60)
    print("  CAR DAMAGE — YOLOv8m-seg EĞİTİM (v3 — 13 sınıf)")
    print("=" * 60)

    data_yaml = check_merged_dataset()
    device    = check_gpu()

    if resume:
        last_weights = RUN_DIR / RUN_NAME / "weights" / "last.pt"
        if not last_weights.exists():
            print("❌ Devam edilecek model bulunamadı. Normal eğitim başlatılıyor...")
            resume = False
        else:
            model_path = str(last_weights)
            print(f"\n  Devam: {last_weights}")

    if not resume:
        if scratch or not PRETRAINED.exists():
            model_path = SCRATCH_MODEL
            print(f"\n  Başlangıç: sıfırdan ({SCRATCH_MODEL})")
        else:
            model_path = str(PRETRAINED)
            print(f"\n  Başlangıç: mevcut best.pt (fine-tune)")

    # Fine-tune ise düşük lr, scratch ise yüksek lr
    is_finetune = (not scratch and not resume and PRETRAINED.exists())
    lr0 = 0.0001 if is_finetune else 0.001

    print(f"\n  Epochs    : {epochs}")
    print(f"  Batch     : {batch}")
    print(f"  lr0       : {lr0}  ({'fine-tune' if is_finetune else 'scratch/resume'})")
    print(f"  Device    : {device}")
    print("=" * 60 + "\n")

    model = YOLO(model_path)

    results = model.train(
        data          = data_yaml,
        epochs        = epochs,
        imgsz         = 640,
        batch         = batch,
        device        = device,
        workers       = 4,
        cache         = True,
        optimizer     = "AdamW",
        lr0           = lr0,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 1 if is_finetune else 3,
        patience      = 15,
        project       = str(RUN_DIR),
        name          = RUN_NAME,
        exist_ok      = True,
        save          = True,
        save_period   = 10,
        val           = True,
        plots         = True,
        task          = "segment",
        verbose       = True,
        amp           = True,
        resume        = resume,
        # Augmentation
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        degrees       = 5.0,
        translate     = 0.1,
        scale         = 0.4,
        flipud        = 0.0,
        fliplr        = 0.5,
        mosaic        = 0.8,
        mixup         = 0.0,
        copy_paste    = 0.1,
    )

    best_model = RUN_DIR / RUN_NAME / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print(f"  ✅ Eğitim Tamamlandı!")
    print(f"  En iyi model: {best_model}")
    print("=" * 60)

    return results


def evaluate():
    best_model = RUN_DIR / RUN_NAME / "weights" / "best.pt"
    if not best_model.exists():
        print(f"❌ Model bulunamadı: {best_model}")
        print("   Önce python train.py ile eğitimi tamamlayın.")
        sys.exit(1)

    data_yaml = check_merged_dataset()
    device    = check_gpu()

    print(f"\n  Model: {best_model}")

    model   = YOLO(str(best_model))
    metrics = model.val(
        data      = data_yaml,
        split     = "test",
        device    = device,
        imgsz     = 640,
        batch     = 8,
        task      = "segment",
        plots     = True,
        save_json = True,
    )

    print("\n" + "=" * 60)
    print("  TEST METRİKLERİ")
    print("=" * 60)
    print(f"  mAP50 (bbox)  : {metrics.box.map50:.4f}")
    print(f"  mAP50 (seg)   : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95 (seg): {metrics.seg.map:.4f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Damage YOLOv8m-seg v3 Eğitim")
    parser.add_argument("--epochs",    type=int,           default=50,  help="Epoch sayısı")
    parser.add_argument("--batch",     type=int,           default=8,   help="Batch boyutu")
    parser.add_argument("--eval-only", action="store_true",             help="Sadece test değerlendirmesi")
    parser.add_argument("--resume",    action="store_true",             help="Kaldığı yerden devam et")
    parser.add_argument("--scratch",   action="store_true",             help="Sıfırdan eğit (best.pt kullanma)")
    args = parser.parse_args()

    if args.eval_only:
        evaluate()
    else:
        train(epochs=args.epochs, batch=args.batch, resume=args.resume, scratch=args.scratch)
