# surgical-robotic-arm
Teleoperated FPV robotic arm with a surgical instrument detection layer.

Master arm (potentiometers) → ESP32 → ESP-NOW → Slave arm (servos via PCA9685) → ESP32-CAM → YOLOv8 detection

---

## Surgical Vision — m2cai16 + YOLOv8

**Dataset:** m2cai16 — 2,811 frames, 7 classes (Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag), VOC2007 XML format

**Pipeline:**
1. Convert VOC XML → YOLO format using official splits (train=1405 / val=843 / test=563)
2. Train YOLOv8n or YOLOv8s
3. Export to ONNX for ROS 2 node

### Sample + Detections

![Sample](https://github.com/user-attachments/assets/beba3b32-4995-47e8-aa59-3c6bf1e25153)
![Det 1](https://github.com/user-attachments/assets/b0f6fba8-0582-4bc9-8bcb-ca7ced1c8c97)
![Det 2](https://github.com/user-attachments/assets/7143d954-0e92-4bd3-b0d1-09a6e471d338)
![Det 3](https://github.com/user-attachments/assets/83e4b6cb-d527-49c5-b93d-1eae3a119f70)
![Det 4](https://github.com/user-attachments/assets/9112d7e6-b22a-4d85-aa7e-7b10eecfac81)

### UI

![UI](https://github.com/user-attachments/assets/a52bd37f-f789-4d71-98e7-a7bacf95cd46)

### Test Video
[Laparoscopic Cholecystectomy HD - Dr. R.K. Mishra](https://www.laparoscopyhospital.com/videos/public/videos/1889/unedited-full-length-laparoscopic-cholecystectomy-hd-video-dr-r-k-mishra-bydEYzknlG)

---

## Hardware
- ESP32 x2, PCA9685, MG996R x3 + SG90 x1, ESP32-CAM

## Setup
```bash
pip install ultralytics opencv-python
python cholecseg8k_train.py
python m2cai16_train.py
```

## References
- [m2cai16](http://camma.u-strasbg.fr/m2cai16/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
