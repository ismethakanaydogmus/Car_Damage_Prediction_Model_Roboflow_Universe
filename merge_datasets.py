"""
İki yeni veri setini birleştirerek merged_dataset/ dizini oluşturur.

Dataset 1 — car damage detection (6 sınıf):
  0: crack, 1: dent, 2: glass shatter, 3: lamp broken, 4: scratch, 5: tire flat

Dataset 2 — Car parts damage (7 sınıf, +6 offset uygulanır):
  6: Bonnet, 7: Bumper, 8: Dickey, 9: Door, 10: Fender, 11: Light, 12: Windshield

Kullanım:
    python merge_datasets.py
"""

import shutil
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent

DATASET1 = BASE_DIR / "car damage detection.v1i.yolov8"
DATASET2 = BASE_DIR / "Car parts damage.v1i.yolov8"
OUTPUT   = BASE_DIR / "merged_dataset"

CLASS_OFFSET = 6  # Dataset2 indisleri bu kadar kaydırılır

CLASSES = [
    # Dataset 1
    "crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat",
    # Dataset 2 (offset uygulanmış)
    "Bonnet", "Bumper", "Dickey", "Door", "Fender", "Light", "Windshield",
]

SPLITS = ["train", "valid", "test"]


def remap_label(src: Path, dst: Path, offset: int) -> None:
    """Label dosyasını okur, sınıf indisine offset ekleyerek yeni dosyaya yazar."""
    lines = src.read_text(encoding="utf-8").strip().splitlines()
    remapped = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        parts[0] = str(int(parts[0]) + offset)
        remapped.append(" ".join(parts))
    dst.write_text("\n".join(remapped) + "\n", encoding="utf-8")


def copy_split(src_root: Path, dst_root: Path, split: str, offset: int) -> int:
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    dst_images = dst_root / split / "images"
    dst_labels = dst_root / split / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    if not src_images.exists():
        return 0

    count = 0
    for img in src_images.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        # Çakışmayı önlemek için kaynak dizin adını prefix olarak ekle
        prefix = src_root.name.replace(" ", "_").replace(".", "_")
        dst_img = dst_images / f"{prefix}__{img.name}"
        shutil.copy2(img, dst_img)

        lbl_src = src_labels / (img.stem + ".txt")
        lbl_dst = dst_labels / f"{prefix}__{img.stem}.txt"

        if lbl_src.exists():
            if offset == 0:
                shutil.copy2(lbl_src, lbl_dst)
            else:
                remap_label(lbl_src, lbl_dst, offset)
        else:
            # Etiket yoksa boş dosya oluştur (negatif örnek)
            lbl_dst.touch()

        count += 1

    return count


def write_yaml(dst_root: Path) -> None:
    cfg = {
        "train": str(dst_root / "train" / "images"),
        "val":   str(dst_root / "valid" / "images"),
        "test":  str(dst_root / "test"  / "images"),
        "nc":    len(CLASSES),
        "names": CLASSES,
        "task":  "segment",
    }
    out = dst_root / "data.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    print(f"  ✓ data.yaml yazıldı: {out}")


def main():
    print("=" * 60)
    print("  VERİ SETİ BİRLEŞTİRME")
    print("=" * 60)

    if OUTPUT.exists():
        print(f"\n⚠  {OUTPUT.name}/ zaten mevcut — siliniyor ve yeniden oluşturuluyor...")
        shutil.rmtree(OUTPUT)

    totals = {s: 0 for s in SPLITS}

    for split in SPLITS:
        n1 = copy_split(DATASET1, OUTPUT, split, offset=0)
        n2 = copy_split(DATASET2, OUTPUT, split, offset=CLASS_OFFSET)
        totals[split] = n1 + n2
        print(f"  {split:5s}: {n1:4d} (dataset1) + {n2:4d} (dataset2) = {n1+n2:4d} görsel")

    write_yaml(OUTPUT)

    print("\n" + "=" * 60)
    print("  ÖZET")
    print("=" * 60)
    print(f"  Toplam sınıf : {len(CLASSES)}")
    for i, c in enumerate(CLASSES):
        print(f"    {i:2d}: {c}")
    print(f"\n  Toplam görsel:")
    for split, n in totals.items():
        print(f"    {split:5s}: {n}")
    print(f"\n  Çıktı dizini : {OUTPUT}")
    print("=" * 60)
    print("\n✅ Birleştirme tamamlandı. Şimdi 'python train.py' komutunu çalıştırabilirsiniz.")


if __name__ == "__main__":
    main()
