# YOLOv8 Fine-Tuning on VisDrone

## Dataset

**VisDrone-DET2019** — aerial/elevated-camera drone imagery.

| Split | Images |
|-------|--------|
| Train | 6 471  |
| Val   | 548    |

**10 classes:** pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

## Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | YOLOv8n (nano) |
| Epochs | 15 |
| Batch size | 16 |
| Image size | 640 × 640 |
| GPU | Tesla T4 (Google Colab) |
| Training time | ~37 minutes |

## Results

| Metric | Baseline (pretrained) | Fine-tuned | Improvement |
|--------|----------------------|------------|-------------|
| mAP50 | 0.0158 | 0.256 | ×16 |
| mAP50-95 | 0.0070 | 0.142 | ×20 |

### Per-Class mAP50

| Class | mAP50 |
|-------|-------|
| car | 0.685 |
| van | 0.288 |
| bus | 0.350 |
| motor | 0.280 |
| pedestrian | 0.263 |
| truck | 0.236 |
| people | 0.195 |
| tricycle | 0.155 |
| awning-tricycle | 0.0715 |
| bicycle | 0.0386 |

## Analysis

Large, well-defined objects (cars, buses, vans) are detected significantly better than small objects (bicycles, awning-tricycles). This is expected for aerial/elevated camera views where small objects occupy very few pixels and lack distinctive texture. More epochs and higher resolution would help small-object detection.

The fine-tuned weights are stored at `src/detection/models/yolov8n_visdrone_best.pt` and are loaded automatically by `YOLODetector` when the file is present.
