# Multimodal Depression Detection (Body + Face)

## 3D Body Pose & 3D Facial Landmark Based Learning

This repository implements a **modular multimodal deep learning framework** for depression-related analysis using:

- **3D Body Pose Sequences**
- **3D Face Landmark Sequences**

It supports:

- Binary depression classification  
- Symptom-based classification (Retardation / Agitation)  
- Rating trend classification (Negative / Neutral / Positive change)  
- Body-only, Face-only, and Fusion training  
- Configurable preprocessing and experiment setups  

---

# 🔎 Overall Workflow

All experiments follow the same pipeline:

```
1) Preprocess raw pose / landmark data → .pt tensors
2) Train model (MSN / Res_Res / TSFFM)
3) Evaluate trained model
```

All commands below are **examples**.  
You can modify:

- `--output_dir`
- `--pose_mode`
- `--symptom_mode`
- `--label_keywords`
- `--label`
- Hyperparameters
- Fusion strategies
- Save paths

---

# 🔧 Environment Setup

```bash
conda create -n depression_env python=3.10
conda activate depression_env
pip install torch torchvision pandas numpy scikit-learn openpyxl tqdm
```

---

# 🧩 Model Architectures

## 1️⃣ MSN

- Body-only architecture
- Single input directory: `--data_dir`
- In rating trend mode: **multi-task** (predicts both `label_n` and `label_p`)
- Simple and efficient baseline

---

## 2️⃣ Res_Res

- Two-stream residual architecture
- Supports:
  - Body-only (`--input_type B`)
  - Face-only (`--input_type F`)
  - Fusion (`--input_type F+B`)
- Fusion via:
  ```
  --fusion avg_logit
  ```
- In rating trend mode: predicts **one target at a time**
  ```
  --label {label_n, label_p}
  ```

---

## 3️⃣ TSFFM

- Two-Stream Feature Fusion Model
- Designed for multimodal feature-level interaction
- Requires:
  - `--body_dir`
  - `--face_dir`
- In rating trend mode:
  ```
  --label {label_n, label_p}
  ```

---

# 🧠 Supported Tasks

---

# 🔵 A) Binary Depression Classification

Healthy vs Depressed

- `0 = healthy`
- `1 = depressed`

Optional filtering:
- `--pose_mode {all, ei, training, coping}`
- `--label_keywords CR CRADK ADK SHAM`

---

## Step A1 — Preprocess Body

```bash
python symptom_classification/prepare_body_data.py \
  --pose_mode coping \
  --label_keywords CRADK ADK \
  --output_dir ./processed_body_binary_coping
```

---

## Step A2 — Preprocess Face

```bash
python symptom_classification/prepare_face_data.py \
  --pose_mode coping \
  --pattern "CRADK|ADK" \
  --output_dir ./processed_face_binary_coping
```

---

## Step A3 — Train

### MSN (Body-only)

```bash
python symptom_classification/MSN/msn_train.py \
  --data_dir ./processed_body_binary_coping \
  --epochs 40 \
  --batch_size 16 \
  --lr 0.001 \
  --save_path ./symptom_classification/MSN/msn_binary_coping.pth
```

---

### Res_Res (Body-only)

```bash
python symptom_classification/Res_Res/resres_train.py \
  --input_type B \
  --body_dir ./processed_body_binary_coping \
  --epochs 40 \
  --batch_size 16 \
  --lr 0.001 \
  --save_path ./symptom_classification/Res_Res/resres_body_binary.pth
```

---

### Res_Res (Face-only)

```bash
python symptom_classification/Res_Res/resres_train.py \
  --input_type F \
  --face_dir ./processed_face_binary_coping \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.0003 \
  --save_path ./symptom_classification/Res_Res/resres_face_binary.pth
```

---

### Res_Res (Fusion)

```bash
python symptom_classification/Res_Res/resres_train.py \
  --input_type F+B \
  --body_dir ./processed_body_binary_coping \
  --face_dir ./processed_face_binary_coping \
  --fusion avg_logit \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.0003 \
  --save_path ./symptom_classification/Res_Res/resres_fusion_binary.pth
```

---

### TSFFM (Fusion)

```bash
python symptom_classification/TSFFM/tsffm_train.py \
  --body_dir ./processed_body_binary_coping \
  --face_dir ./processed_face_binary_coping \
  --epochs 40 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_path ./symptom_classification/TSFFM/tsffm_binary.pth
```

---

## Step A4 — Evaluate

```bash
python symptom_classification/MSN/msn_eval.py \
  --data_dir ./processed_body_binary_coping \
  --model_path ./symptom_classification/MSN/msn_binary_coping.pth
```

```bash
python symptom_classification/Res_Res/resres_eval.py \
  --input_type F+B \
  --body_dir ./processed_body_binary_coping \
  --face_dir ./processed_face_binary_coping \
  --model_path ./symptom_classification/Res_Res/resres_fusion_binary.pth
```

```bash
python symptom_classification/TSFFM/tsffm_eval.py \
  --body_dir ./processed_body_binary_coping \
  --face_dir ./processed_face_binary_coping \
  --model_path ./symptom_classification/TSFFM/tsffm_binary.pth
```

---

# 🟡 B) Symptom Classification

Retardation or Agitation (binary per symptom)

---

## Step B1 — Generate Labels

```bash
python symptom_classification/get_retardation_agitation_labels.py
```

Generates:
- `retardation_labels.csv`
- `agitation_labels.csv`

---

## Step B2 — Preprocess

```bash
python symptom_classification/prepare_body_data_symptom_modes.py \
  --symptom_mode agitation \
  --pose_mode coping \
  --output_dir ./processed_body_agitation_coping
```

```bash
python symptom_classification/prepare_face_data_symptom_modes.py \
  --symptom_mode agitation \
  --pose_mode coping \
  --output_dir ./processed_face_agitation_coping
```

---

## Step B3 — Train (Example: Fusion)

```bash
python symptom_classification/Res_Res/resres_train.py \
  --input_type F+B \
  --body_dir ./processed_body_agitation_coping \
  --face_dir ./processed_face_agitation_coping \
  --fusion avg_logit \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.0003 \
  --save_path ./symptom_classification/Res_Res/resres_agitation.pth
```

---

# 🔴 C) Rating Trend Classification

Predicts change in rating over time:

- `0 = negative`
- `1 = neutral`
- `2 = positive`

---

## Step C1 — Compute Rating Differences

```bash
python rating_trend/calculate_rating_diffs.py
```

---

## Step C2 — Preprocess Trials

```bash
python rating_trend/prepare_body_trends.py \
  --output_dir ./processed_body_trends_exp1
```

```bash
python rating_trend/prepare_face_trends.py \
  --output_dir ./processed_face_trends_exp1
```

---

## Step C3 — Train

### MSN (Multi-task)

```bash
python rating_trend/MSN/msn_train.py \
  --data_dir ./processed_body_trends_exp1 \
  --epochs 30 \
  --batch_size 16 \
  --lr 3e-4 \
  --save_path ./rating_trend/MSN/msn_trend_exp1.pth
```

(No `--label` required.)

---

### Res_Res (Single Target)

```bash
python rating_trend/Res_Res/resres_train.py \
  --input_type F+B \
  --body_dir ./processed_body_trends_exp1 \
  --face_dir ./processed_face_trends_exp1 \
  --label label_n \
  --fusion avg_logit \
  --epochs 60 \
  --batch_size 16 \
  --lr 3e-4 \
  --save_path ./rating_trend/Res_Res/resres_trend_label_n.pth
```

---

### TSFFM (Single Target)

```bash
python rating_trend/TSFFM/tsffm_train.py \
  --body_dir ./processed_body_trends_exp1 \
  --face_dir ./processed_face_trends_exp1 \
  --label label_p \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-4 \
  --save_path ./rating_trend/TSFFM/tsffm_trend_label_p.pth
```

---

# 📁 Repository Structure

```
symptom_classification/
    labels/
    MSN/
    Res_Res/
    TSFFM/
    prepare_*.py

rating_trend/
    MSN/
    Res_Res/
    TSFFM/
    calculate_rating_diffs.py
    prepare_*.py
```

---

# 🧠 Best Practices

- Use a new `processed_*` directory per experiment
- Keep preprocessing and training directories consistent
- Log hyperparameters
- Start with small epochs before long training runs
- Use separate folders for binary, symptom, and trend experiments