"""
cholecseg8k_train.py
====================
YOLOv8 surgical instrument detection pipeline on CholecSeg8k.

Dataset : CholecSeg8k (newslab/cholecseg8k on Kaggle)
          ~8000 frames from 17 cholecystectomy videos, 13 semantic classes
Targets : grasper (pixel 23), l_hook_electrocautery (pixel 27)
Model   : YOLOv8n trained from COCO pretrained weights
Output  : best.pt + ONNX export

Pipeline
--------
1. Download CholecSeg8k from Kaggle
2. Scan dataset — count instrument presence per frame
3. Convert segmentation masks -> YOLO bounding box labels
4. Train/val split (80/20, random seed=42)
5. Train YOLOv8n
6. Export to ONNX (opset 12)

Usage (Google Colab)
--------------------
    Run cells top to bottom. Results saved to Google Drive under
    /MyDrive/surgical_yolo/

Requirements
------------
    pip install ultralytics kaggle opencv-python-headless matplotlib tqdm
"""

# ============================================================
# SECTION 1 — SETUP
# ============================================================
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Mount Google Drive (Colab only)
from google.colab import drive, files
drive.mount('/content/drive')

DRIVE_BASE   = '/content/drive/MyDrive/surgical_yolo'
DATA_ROOT    = f'{DRIVE_BASE}/raw_data'
OUT_IMAGES   = f'{DRIVE_BASE}/yolo_dataset/images'
OUT_LABELS   = f'{DRIVE_BASE}/yolo_dataset/labels'

os.makedirs(DRIVE_BASE, exist_ok=True)
os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)
print(f"Project folder ready: {DRIVE_BASE}")


# ============================================================
# SECTION 2 — DOWNLOAD DATASET (run once)
# ============================================================
def setup_kaggle_and_download():
    """Upload kaggle.json credentials and download CholecSeg8k."""
    print("Upload your kaggle.json file...")
    uploaded = files.upload()
    os.makedirs('/root/.kaggle', exist_ok=True)
    os.rename('/content/kaggle.json', '/root/.kaggle/kaggle.json')
    os.chmod('/root/.kaggle/kaggle.json', 0o600)
    print("Kaggle credentials set.")

    print("Downloading CholecSeg8k (~3 GB)...")
    os.system(f'kaggle datasets download -d newslab/cholecseg8k -p {DATA_ROOT} --unzip')
    print("Dataset downloaded.")

# Uncomment to run on first use:
# setup_kaggle_and_download()


# ============================================================
# SECTION 3 — CLASS MAPPING
# ============================================================
# Official CholecSeg8k pixel-value -> class-name mapping
CLASS_MAP = {
    0:  'background',
    11: 'black_background',
    12: 'abdominal_wall',
    13: 'liver',
    21: 'gastrointestinal_tract',
    22: 'fat',
    23: 'grasper',               # INSTRUMENT
    24: 'connective_tissue',
    25: 'blood',
    26: 'cystic_duct',
    27: 'l_hook_electrocautery', # INSTRUMENT
    28: 'gallbladder',
    29: 'hepatic_vein',
    31: 'liver_ligament',
}

# Classes used for detection (pixel value -> YOLO class id)
INSTRUMENT_CLASSES = {23: 0, 27: 1}
CLASS_NAMES        = ['grasper', 'l_hook_electrocautery']


# ============================================================
# SECTION 4 — DATASET STATISTICS
# ============================================================
def scan_instrument_presence(data_root):
    """Count frames containing each instrument class."""
    instrument_frames = 0
    total_frames      = 0

    for video in sorted(os.listdir(data_root)):
        video_path = os.path.join(data_root, video)
        if not os.path.isdir(video_path):
            continue
        for clip in os.listdir(video_path):
            clip_path = os.path.join(video_path, clip)
            for f in os.listdir(clip_path):
                if not (f.endswith('_mask.png')
                        and 'color' not in f
                        and 'watershed' not in f):
                    continue
                total_frames += 1
                mask = np.array(Image.open(os.path.join(clip_path, f)))
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                if any(cls in np.unique(mask) for cls in INSTRUMENT_CLASSES):
                    instrument_frames += 1

    print(f"Total frames       : {total_frames}")
    print(f"Frames w/ instruments: {instrument_frames} "
          f"({100 * instrument_frames / total_frames:.1f}%)")
    return total_frames, instrument_frames

total_frames, instrument_frames = scan_instrument_presence(DATA_ROOT)


# ============================================================
# SECTION 5 — VISUALISE A SAMPLE (optional)
# ============================================================
def visualise_sample(clip_path, base_name, save_path=None):
    """Display raw image, annotation mask, and color mask side by side."""
    img        = Image.open(f'{clip_path}/{base_name}_endo.png')
    mask       = Image.open(f'{clip_path}/{base_name}_endo_mask.png')
    color_mask = Image.open(f'{clip_path}/{base_name}_endo_color_mask.png')

    mask_np = np.array(mask)
    print(f"Image size : {img.size}")
    print(f"Classes present: {np.unique(mask_np)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);             axes[0].set_title('Raw Image');        axes[0].axis('off')
    axes[1].imshow(mask_np, cmap='tab20'); axes[1].set_title('Annotation Mask'); axes[1].axis('off')
    axes[2].imshow(color_mask);      axes[2].set_title('Color Mask');       axes[2].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# Example usage:
# visualise_sample(
#     '/content/data/video01/video01_00080',
#     'frame_100',
#     save_path=f'{DRIVE_BASE}/sample_viz.png'
# )


# ============================================================
# SECTION 6 — MASK -> YOLO LABEL CONVERSION
# ============================================================
IMG_W = 854
IMG_H = 480

def mask_to_yolo_boxes(mask_np, img_w=IMG_W, img_h=IMG_H):
    """
    Convert a segmentation mask to YOLO bounding box lines.

    Args:
        mask_np : numpy array (H, W) or (H, W, C) — single-channel mask
        img_w   : image width  in pixels
        img_h   : image height in pixels

    Returns:
        List of strings, each in YOLO format:
        "<class_id> <cx> <cy> <w> <h>"  (all normalised to [0, 1])
    """
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    boxes = []
    for pixel_val, class_id in INSTRUMENT_CLASSES.items():
        region = (mask_np == pixel_val).astype(np.uint8)
        if region.sum() < 100:          # skip noise / tiny artefacts
            continue
        ys, xs   = np.where(region)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cx = (x_min + x_max) / 2 / img_w
        cy = (y_min + y_max) / 2 / img_h
        w  = (x_max - x_min) / img_w
        h  = (y_max - y_min) / img_h
        boxes.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return boxes


def convert_dataset(data_root, out_images, out_labels):
    """Walk CholecSeg8k tree and write YOLO labels for all instrument frames."""
    converted = 0
    skipped   = 0

    for video in tqdm(sorted(os.listdir(data_root)), desc="Videos"):
        video_path = os.path.join(data_root, video)
        if not os.path.isdir(video_path):
            continue
        for clip in sorted(os.listdir(video_path)):
            clip_path = os.path.join(video_path, clip)
            if not os.path.isdir(clip_path):
                continue
            for f in sorted(os.listdir(clip_path)):
                if not (f.endswith('_endo.png') and 'mask' not in f):
                    continue
                base      = f.replace('_endo.png', '')
                img_path  = os.path.join(clip_path, f)
                mask_path = os.path.join(clip_path, f'{base}_endo_mask.png')
                if not os.path.exists(mask_path):
                    continue

                mask_np = np.array(Image.open(mask_path))
                boxes   = mask_to_yolo_boxes(mask_np)
                if not boxes:
                    skipped += 1
                    continue

                Image.open(img_path).save(
                    os.path.join(out_images, f'{video}_{base}.png'))
                with open(os.path.join(out_labels, f'{video}_{base}.txt'), 'w') as lf:
                    lf.write('\n'.join(boxes))
                converted += 1

    print(f"Converted : {converted} frames with instruments")
    print(f"Skipped   : {skipped}   frames without instruments")

convert_dataset(DATA_ROOT, OUT_IMAGES, OUT_LABELS)


# ============================================================
# SECTION 7 — TRAIN / VAL SPLIT (80/20)
# ============================================================
SPLIT_BASE = f'{DRIVE_BASE}/yolo_split'

for split in ['train', 'val']:
    os.makedirs(f'{SPLIT_BASE}/images/{split}', exist_ok=True)
    os.makedirs(f'{SPLIT_BASE}/labels/{split}', exist_ok=True)

random.seed(42)
all_images = sorted([f for f in os.listdir(OUT_IMAGES) if f.endswith('.png')])
random.shuffle(all_images)
split_idx   = int(0.8 * len(all_images))
train_files = all_images[:split_idx]
val_files   = all_images[split_idx:]
print(f"Train: {len(train_files)} | Val: {len(val_files)}")

for split, flist in [('train', train_files), ('val', val_files)]:
    for fname in flist:
        shutil.copy(f'{OUT_IMAGES}/{fname}',
                    f'{SPLIT_BASE}/images/{split}/{fname}')
        lbl = fname.replace('.png', '.txt')
        shutil.copy(f'{OUT_LABELS}/{lbl}',
                    f'{SPLIT_BASE}/labels/{split}/{lbl}')

# Write data.yaml
yaml_content = (
    f"path: {SPLIT_BASE}\n"
    f"train: images/train\n"
    f"val: images/val\n"
    f"nc: {len(CLASS_NAMES)}\n"
    f"names: {CLASS_NAMES}\n"
)
yaml_path = f'{DRIVE_BASE}/dataset.yaml'
with open(yaml_path, 'w') as f:
    f.write(yaml_content)
print("dataset.yaml written.")


# ============================================================
# SECTION 8 — TRAIN YOLOv8n
# ============================================================
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # pretrained COCO weights

model.train(
    data       = yaml_path,
    epochs     = 50,
    imgsz      = 640,
    batch      = 16,
    device     = 0,
    project    = f'{DRIVE_BASE}/runs',
    name       = 'surgical_yolo_v1',
    # Augmentation
    mosaic     = 1.0,
    flipud     = 0.3,
    fliplr     = 0.5,
    degrees    = 10.0,
    hsv_h      = 0.015,
    hsv_s      = 0.5,
    hsv_v      = 0.3,
)


# ============================================================
# SECTION 9 — EXPORT TO ONNX
# ============================================================
best_weights = f'{DRIVE_BASE}/runs/surgical_yolo_v1/weights/best.pt'
model        = YOLO(best_weights)

onnx_path = model.export(
    format   = 'onnx',
    imgsz    = 640,
    opset    = 12,       # safe for onnxruntime / TensorRT
    simplify = True,
    dynamic  = False,    # fixed batch=1 for inference node
)

onnx_dst = f'{DRIVE_BASE}/surgical_yolo_v1.onnx'
shutil.copy2(onnx_path, onnx_dst)
print(f"ONNX exported to: {onnx_dst}")
