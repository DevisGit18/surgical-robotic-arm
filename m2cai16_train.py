"""
m2cai16_train.py
================
YOLOv8 surgical instrument detection pipeline on M2CAI16.

Dataset : M2CAI16 Tool Locations
          2811 annotated frames, VOC2007 XML format
          7 classes: Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag
          Official splits: train=1405 / val=843 / test=563
Model   : YOLOv8s trained from COCO pretrained weights
Output  : best.pt + ONNX export + annotated video inference

Pipeline
--------
1. Mount Drive, configure paths
2. Convert VOC XML -> YOLO txt labels (using official ImageSets splits)
3. Write data.yaml
4. Label sanity check
5. Train YOLOv8s
6. Validate on held-out test set
7. Export to ONNX (opset 12)
8. Run inference on a sample video

Usage (Google Colab)
--------------------
    Place the m2cai16-tool-locations dataset folder on Google Drive, then
    run cells top to bottom.

Requirements
------------
    pip install ultralytics onnxruntime-gpu opencv-python-headless tqdm pyyaml
"""

# ============================================================
# SECTION 1 — SETUP
# ============================================================
import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm
from ultralytics import YOLO

# Mount Google Drive (Colab only)
from google.colab import drive
drive.mount('/content/drive')

# Source dataset (read-only, on Drive)
SRC      = Path('/content/drive/MyDrive/m2cai16-tool-locations/m2cai16-tool-locations')
ANN_DIR  = SRC / 'Annotations'
IMG_DIR  = SRC / 'JPEGImages'
SETS_DIR = SRC / 'ImageSets' / 'Main'

# Destination on local SSD (fast I/O during training)
DEST     = Path('/content/m2cai16_yolo')

for split in ['train', 'val', 'test']:
    (DEST / split / 'images').mkdir(parents=True, exist_ok=True)
    (DEST / split / 'labels').mkdir(parents=True, exist_ok=True)

print('Paths ready.')
print(f'  Source images : {len(list(IMG_DIR.glob("*.jpg")))} JPEGs')
print(f'  Source XMLs   : {len(list(ANN_DIR.glob("*.xml")))} XMLs')


# ============================================================
# SECTION 2 — CLASS MAPPING
# ============================================================
# Confirmed from class_list.txt — order is index 0..6
CLASS_NAMES = ['Grasper', 'Bipolar', 'Hook', 'Scissors',
               'Clipper', 'Irrigator', 'SpecimenBag']
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


# ============================================================
# SECTION 3 — VOC XML -> YOLO LABEL CONVERSION
# ============================================================
def voc_xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """
    Parse one VOC XML annotation file and return YOLO label lines.

    Args:
        xml_path : path to the .xml annotation file
        img_w    : image width  in pixels
        img_h    : image height in pixels

    Returns:
        List of strings in YOLO format:
        "<class_id> <cx> <cy> <w> <h>"  (all normalised to [0, 1])
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        name = obj.findtext('name', '').strip()
        if name not in CLASS_TO_ID:
            print(f'  Unknown class "{name}" in {xml_path.name} — skipped')
            continue
        cls_id = CLASS_TO_ID[name]
        bb   = obj.find('bndbox')
        xmin = float(bb.findtext('xmin'))
        ymin = float(bb.findtext('ymin'))
        xmax = float(bb.findtext('xmax'))
        ymax = float(bb.findtext('ymax'))
        # Clamp to image bounds
        xmin = max(0.0, min(xmin, img_w))
        ymin = max(0.0, min(ymin, img_h))
        xmax = max(0.0, min(xmax, img_w))
        ymax = max(0.0, min(ymax, img_h))
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    return lines


def convert_dataset(sets_dir, img_dir, ann_dir, dest):
    """Convert all splits from VOC XML to YOLO txt format."""
    stats = {s: {'imgs': 0, 'labels': 0, 'skipped': 0}
             for s in ['train', 'val', 'test']}

    for split in ['train', 'val', 'test']:
        split_file = sets_dir / f'{split}.txt'
        stems = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
        print(f'\nProcessing {split}: {len(stems)} entries')

        for stem in tqdm(stems, desc=split):
            img_src = img_dir / f'{stem}.jpg'
            xml_src = ann_dir / f'{stem}.xml'
            img_dst = dest / split / 'images' / f'{stem}.jpg'
            lbl_dst = dest / split / 'labels' / f'{stem}.txt'

            if not img_src.exists() or not xml_src.exists():
                stats[split]['skipped'] += 1
                print(f'  Missing: {stem}')
                continue

            # Read image dimensions from XML (avoids opening every JPEG)
            tree    = ET.parse(xml_src)
            size_el = tree.getroot().find('size')
            img_w   = int(size_el.findtext('width'))
            img_h   = int(size_el.findtext('height'))

            yolo_lines = voc_xml_to_yolo(xml_src, img_w, img_h)
            shutil.copy2(img_src, img_dst)
            stats[split]['imgs'] += 1
            lbl_dst.write_text('\n'.join(yolo_lines))
            if yolo_lines:
                stats[split]['labels'] += 1

    print('\n=== CONVERSION SUMMARY ===')
    for split, s in stats.items():
        print(f'  {split:5s}: {s["imgs"]} images, '
              f'{s["labels"]} with labels, {s["skipped"]} skipped')

convert_dataset(SETS_DIR, IMG_DIR, ANN_DIR, DEST)


# ============================================================
# SECTION 4 — WRITE data.yaml
# ============================================================
data_cfg = {
    'path' : str(DEST),
    'train': 'train/images',
    'val'  : 'val/images',
    'test' : 'test/images',
    'nc'   : len(CLASS_NAMES),
    'names': CLASS_NAMES,
}

yaml_path = DEST / 'data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

print('data.yaml:')
print(yaml_path.read_text())

for split in ['train', 'val', 'test']:
    n_imgs = len(list((DEST / split / 'images').glob('*.jpg')))
    n_lbls = len(list((DEST / split / 'labels').glob('*.txt')))
    print(f'  {split}: {n_imgs} images, {n_lbls} labels')


# ============================================================
# SECTION 5 — LABEL SANITY CHECK
# ============================================================
def sanity_check_labels(dest, class_names, n_samples=10):
    """Spot-check random label files for format correctness."""
    lbl_files = list((dest / 'train' / 'labels').glob('*.txt'))
    samples   = random.sample(lbl_files, min(n_samples, len(lbl_files)))

    print(f'=== LABEL SANITY CHECK ({n_samples} random train labels) ===')
    for lf in samples:
        lines = lf.read_text().strip().splitlines()
        print(f'\n{lf.name} ({len(lines)} objects):')
        for line in lines:
            parts  = line.split()
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            print(f'  {class_names[cls_id]}({cls_id})  '
                  f'cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}')
            assert 0 <= cx <= 1 and 0 <= cy <= 1, f'cx/cy out of bounds: {cx},{cy}'
            assert 0 < w <= 1  and 0 < h  <= 1,  f'w/h out of bounds: {w},{h}'
    print('\nAll sampled labels are valid.')

sanity_check_labels(DEST, CLASS_NAMES)


# ============================================================
# SECTION 6 — TRAIN YOLOv8s
# ============================================================
DRIVE_OUT = '/content/drive/MyDrive/surgical_yolo'

model = YOLO('yolov8s.pt')   # pretrained COCO weights (~22 MB)

results = model.train(
    data          = str(yaml_path),
    epochs        = 100,
    imgsz         = 640,
    batch         = 16,        # fits T4 (15 GB) comfortably with yolov8s
    device        = 0,
    workers       = 4,
    project       = DRIVE_OUT,
    name          = 'm2cai16_yolov8s',
    exist_ok      = True,
    # Optimizer
    optimizer     = 'AdamW',
    lr0           = 0.001,
    lrf           = 0.01,
    warmup_epochs = 3,
    # Augmentation
    fliplr        = 0.5,
    mosaic        = 1.0,
    scale         = 0.5,
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,
    # Checkpointing
    save_period   = 10,
    plots         = True,
    verbose       = True,
)

print(f'\nTraining complete. Best weights: {results.save_dir}/weights/best.pt')


# ============================================================
# SECTION 7 — VALIDATE ON TEST SET
# ============================================================
best_weights = Path(f'{DRIVE_OUT}/m2cai16_yolov8s/weights/best.pt')
model        = YOLO(str(best_weights))

metrics = model.val(
    data   = str(yaml_path),
    split  = 'test',
    imgsz  = 640,
    batch  = 16,
    device = 0,
    conf   = 0.25,
    iou    = 0.5,
)

print('\n=== TEST SET RESULTS ===')
print(f'mAP@0.5      : {metrics.box.map50:.4f}')
print(f'mAP@0.5:0.95 : {metrics.box.map:.4f}')
print('\nPer-class AP@0.5:')
for name, ap in zip(CLASS_NAMES, metrics.box.ap50):
    print(f'  {name:<15s}: {ap:.4f}')


# ============================================================
# SECTION 8 — EXPORT TO ONNX
# ============================================================
onnx_path = model.export(
    format   = 'onnx',
    imgsz    = 640,
    opset    = 12,       # safe for onnxruntime / TensorRT
    simplify = True,
    dynamic  = False,    # fixed batch=1 for inference node
)

onnx_dst = Path(f'{DRIVE_OUT}/m2cai16_yolov8s.onnx')
shutil.copy2(onnx_path, onnx_dst)
print(f'ONNX exported to: {onnx_dst}')
print('Load in inference node with: onnxruntime.InferenceSession(str(onnx_path))')


# ============================================================
# SECTION 9 — VIDEO INFERENCE (ONNX)
# ============================================================
ONNX_PATH   = str(onnx_dst)
VIDEO_PATH  = '/content/input.webm'          # place input video here
OUTPUT_PATH = '/content/output.mp4'
IMG_SIZE    = 640
CONF_THRES  = 0.25
IOU_THRES   = 0.45


def preprocess_frame(frame, img_size):
    """Letterbox a BGR frame to img_size x img_size and normalise to [0,1]."""
    h, w   = frame.shape[:2]
    scale  = img_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas  = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    img = canvas[:, :, ::-1].transpose(2, 0, 1)          # BGR->RGB, HWC->CHW
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0
    return img, scale


def postprocess_detections(outputs, scale, conf_thres, iou_thres):
    """
    Decode raw YOLOv8 ONNX output and apply NMS.

    Args:
        outputs    : list of numpy arrays from onnxruntime
        scale      : letterbox scale factor used in preprocess_frame
        conf_thres : confidence threshold
        iou_thres  : NMS IoU threshold

    Returns:
        List of (box, score, class_id) tuples — boxes in original image coords.
    """
    preds = np.transpose(outputs[0], (0, 2, 1))[0]   # (1,84,8400) -> (8400,84)

    boxes, scores, class_ids = [], [], []
    for pred in preds:
        cls_scores = pred[4:]
        cls_id     = int(np.argmax(cls_scores))
        score      = float(cls_scores[cls_id])
        if score < conf_thres:
            continue
        cx, cy, w, h = pred[:4]
        x1 = int((cx - w / 2) / scale)
        y1 = int((cy - h / 2) / scale)
        x2 = int((cx + w / 2) / scale)
        y2 = int((cy + h / 2) / scale)
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(cls_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(indices) == 0:
        return []
    return [(boxes[i], scores[i], class_ids[i]) for i in indices.flatten()]


def run_video_inference(onnx_path, video_path, output_path,
                        class_names, img_size=640,
                        conf_thres=0.25, iou_thres=0.45):
    """Run frame-by-frame ONNX inference and write annotated output video."""
    session    = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video: {w}x{h} @ {fps} fps')

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img, scale  = preprocess_frame(frame, img_size)
        outputs     = session.run(None, {input_name: img})
        detections  = postprocess_detections(outputs, scale, conf_thres, iou_thres)

        for box, score, cls_id in detections:
            x1, y1, x2, y2 = box
            label = (class_names[cls_id] if cls_id < len(class_names)
                     else f'unknown_{cls_id}')
            label = f'{label} {score:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f'Processed {frame_count} frames...')

    cap.release()
    out.release()
    print(f'Done — {frame_count} frames written to: {output_path}')


# Run inference on input video
run_video_inference(
    onnx_path   = ONNX_PATH,
    video_path  = VIDEO_PATH,
    output_path = OUTPUT_PATH,
    class_names = CLASS_NAMES,
)
