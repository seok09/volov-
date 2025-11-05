# volov-
## 🧩 목차

1. [YOLOv11 소개](#yolov11-소개)
2. [YOLOv8 vs YOLOv11 비교](#yolov8-vs-yolov11-비교)
3. [성능 지표 그래프](#성능-지표-그래프)
4. [YOLOv11 모델 다운로드](#yolov11-모델-다운로드)
5. [YOLO 핵심 용어 정리](#yolo-핵심-용어-정리)
6. [설치 및 실행 예시](#설치-및-실행-예시)
7. [결과 예시](#결과-예시)
8. [참고 자료](#참고-자료)
9. [추가 학습 계획](#추가-학습-계획)

---

## 🔍 YOLOv11 소개

**YOLO (You Only Look Once)** 는 한 번의 신경망 연산으로  
이미지 속 객체의 위치와 종류를 동시에 예측하는 **실시간 객체 탐지 모델**입니다.  

**YOLOv11**은 Ultralytics에서 2024년에 발표한 최신 버전으로,  
이전 세대인 YOLOv8보다 **정확도, 속도, 효율성**이 모두 향상되었습니다.

> ✅ YOLOv11은 Detection, Segmentation, Classification, Pose Estimation까지 지원하는  
> 통합형 비전 모델입니다.

---

## ⚖️ YOLOv8 vs YOLOv11 비교

| 항목 | YOLOv8 | YOLOv11 |
|------|--------|----------|
| **출시 시기** | 2023년 초 | 2024년 말 |
| **개발사** | Ultralytics | Ultralytics |
| **백본(Backbone)** | CSPDarknet 기반 | **C2f-Darknet** (효율적 특징 추출) |
| **Neck 구조** | PAN/FPN | 개선된 Lightweight Fusion |
| **탐지 Head** | Decoupled Detection Head | **Unified Efficient Head** |
| **지원 작업(Task)** | Detection / Segmentation / Classification | **Detection / Segmentation / Classification / Pose Estimation** |
| **정확도 (mAP)** | 높음 | **YOLOv8 대비 +2~4% 향상** |
| **속도 (FPS)** | 빠름 | **더 빠름 (최적화된 연산 구조)** |
| **모델 크기** | n, s, m, l, x | n, s, m, l, x + custom 지원 |
| **Export 기능** | ONNX, TorchScript, TensorRT | **다양한 포맷 지원 (ONNX, TensorRT, CoreML 등)** |
| **활용 분야** | 일반 객체 탐지 | **산업, 의료, 로봇, IoT 등 확장** |

> 💡 **요약:** YOLOv11은 YOLOv8의 속도와 정확도를 모두 개선한 최신 버전으로  
> 실시간 추론 환경에서 최고의 성능을 보입니다.

---

## 📊 성능 지표 그래프

아래 그래프는 **COCO 데이터셋 기준 YOLOv11의 성능(mAP)과 Latency(지연 시간)**을 나타냅니다.  

![YOLOv11 성능 그래프](./performance.png)

> 그래프 출처: Ultralytics YOLO 공식 문서  
> X축: Latency (T4 TensorRT10 FP16, ms/img)  
> Y축: COCO mAP 50-95 (정확도)

---

## 📦 YOLOv11 모델 다운로드

아래 링크를 클릭하면 각 모델의 학습된 가중치 파일을 다운로드할 수 있습니다.  

| 모델 이름 | 설명 | 다운로드 |
|------------|------|-----------|
| **YOLOv11n** | 초경량 Nano 모델 (가장 빠름) | [⬇️ YOLOv11n 다운로드](https://github.com/ultralytics/assets/releases/download/v11.0/yolov11n.pt) |
| **YOLOv11s** | Small 모델 (균형형) | [⬇️ YOLOv11s 다운로드](https://github.com/ultralytics/assets/releases/download/v11.0/yolov11s.pt) |
| **YOLOv11m** | Medium 모델 (정확도 우선) | [⬇️ YOLOv11m 다운로드](https://github.com/ultralytics/assets/releases/download/v11.0/yolov11m.pt) |
| **YOLOv11l** | Large 모델 (고정밀) | [⬇️ YOLOv11l 다운로드](https://github.com/ultralytics/assets/releases/download/v11.0/yolov11l.pt) |
| **YOLOv11x** | Extra-Large 모델 (최고 성능) | [⬇️ YOLOv11x 다운로드](https://github.com/ultralytics/assets/releases/download/v11.0/yolov11x.pt) |

---

## 📘 YOLO 핵심 용어 정리

| 용어 | 설명 | 예시 |
|------|------|------|
| **Object Detection** | 이미지 속 객체를 탐지 및 분류 | 사람, 자동차 탐지 |
| **Bounding Box (BBox)** | 객체 위치를 표시하는 사각형 | (x, y, w, h) 좌표 |
| **Confidence** | 탐지된 객체일 확률 (0~1) | 0.93 → 93% 신뢰도 |
| **Class** | 객체의 종류 | person, car, dog 등 |
| **IoU (Intersection over Union)** | 예측 박스와 실제 박스의 겹치는 정도 | IoU=0.85 |
| **mAP (mean Average Precision)** | 평균 정밀도 지표 | 높을수록 좋음 |
| **NMS (Non-Max Suppression)** | 중복된 박스 제거 | 겹치는 탐지 제거 |
| **Anchor Box** | 미리 정의된 박스 크기 | YOLOv3~v7에서 사용 |
| **Backbone** | 특징 추출 네트워크 | Darknet, C2f-Darknet |
| **Neck** | 다양한 스케일 특징 결합 | FPN, PAN |
| **Head** | 최종 예측 출력 | 클래스, 좌표 예측 |
| **Epoch** | 데이터셋 1회 학습 단위 | 50 epochs |
| **Batch Size** | 한 번에 학습하는 데이터 수 | batch=16 |
| **Learning Rate (LR)** | 학습 속도 조절값 | 0.001 |
| **Data Augmentation** | 데이터 변형 | flip, rotate, crop |

---

## ⚙️ 설치 및 실행 예시

YOLOv11은 Ultralytics 패키지를 이용해 간단히 설치할 수 있습니다.

```bash
# YOLO 설치
pip install ultralytics

# 버전 확인
yolo version

# 테스트 실행
