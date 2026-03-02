# Multimodal Depression Detection (Body + Face)

## 3D Body Pose & 3D Facial Landmark Based Learning

This repository implements a **modular multimodal deep learning framework** for depression-related analysis using:

- **3D Body Pose Sequences**
- **3D Face Landmark Sequences**

The framework supports:

- Multiple clinical prediction tasks  
- Unimodal (Body-only / Face-only) training  
- Multimodal fusion (Body + Face)  
- Flexible preprocessing and training configurations  
- Dynamic output directories for preprocessing scripts (`--output_dir`)

---

# 🔎 Overview

The repository is organized around **three prediction settings**:

1. **Binary Depression Classification**
2. **Symptom Classification (Retardation / Agitation)**
3. **Rating Trend Classification (Negative / Neutral / Positive change)**

For each setting, the workflow is always:

```
1) Preprocess body/face → .pt tensors
2) Train a model (MSN / Res_Res / TSFFM)
3) Evaluate the trained model
```

⚠ **All commands below are examples.**  
You can modify:
- Output directories
- Pose modes
- Target labels (`label_n`, `label_p`)
- Hyperparameters (epochs, batch size, lr)
- Fusion strategies
- Model save paths

---

# 🧠 Supported Tasks

## 1) Binary Depression Classification

**Healthy vs Depressed**

- Labels:
  - `0 = healthy`
  - `1 = depressed`
- Optional group filtering via regex keywords (e.g., `CRADK | ADK | CR | SHAM`)
- Supports pose filtering modes: `all`, `ei`, `training`, `coping`

---

## 2) Symptom Classification

**Retardation vs Agitation** (binary classification per symptom)

- Labels:
  - `0 = healthy`
  - `1 = depressed`
- Separate label CSVs:
  - `retardation_labels.csv`
  - `agitation_labels.csv`

---

## 3) Rating Trend Classification

**Change in rating over time**

- Classes:
  - `0 = negative`
  - `1 = neutral`
  - `2 = positive`
- Targets:
  - `label_n`
  - `label_p`

Rating trend preprocessing generates **trial-level** samples:

```
{
  "data": Tensor [3, T, J],
  "label_n": int,
  "label_p": int,
  "trial": str,
  "subject_id": str
}
```

---

# 📁 Repository Structure

```
symptom_classification/
├── labels/
├── MSN/
├── Res_Res/
├── TSFFM/
├── prepare_body_data.py
├── prepare_face_data.py
├── prepare_body_data_symptom_modes.py
├── prepare_face_data_symptom_modes.py
├── get_retardation_agitation_labels.py

rating_trend/
├── MSN/
├── Res_Res/
├── TSFFM/
├── calculate_rating_diffs.py
├── prepare_body_trends.py
├── prepare_face_trends.py
├── rating_differences.csv
```

---

# 🎭 Pose Modes

Used in preprocessing scripts:

```
--pose_mode {all, ei, training, coping}
```

| Mode       | Description            |
|------------|------------------------|
| `all`      | Use all pose segments  |
| `ei`       | Only EI segments       |
| `training` | Only training segments |
| `coping`   | Only coping segments   |

---

# 🛠 Preprocessing

Preprocessing converts raw pose/landmark data into fixed-length PyTorch tensors (`.pt`) used during training.

Each preprocessing script typically:
1. Loads raw JSON / CSV sources
2. Filters segments using `--pose_mode` (if applicable)
3. Matches participants/trials with labels
4. Normalizes coordinates (relative to first joint/landmark)
5. Pads/truncates sequences to `T=300`
6. Saves `.pt` tensors into `--output_dir`

---

## 📦 Output Format

### Binary / Symptom Mode

```
{
  "data": Tensor [3, T, J],
  "label": int
}
```

### Rating Trend Mode

```
{
  "data": Tensor [3, T, J],
  "label_n": int,
  "label_p": int,
  "trial": str,
  "subject_id": str
}
```

---

## 📁 Output Directory (`--output_dir`)

Most preprocessing scripts accept:

```
--output_dir PATH
```

Training must point to the same directory using:
- `--data_dir` (MSN)
- `--body_dir` / `--face_dir` (Res_Res, TSFFM)

💡 Best practice: create a **new output directory per configuration** (task + pose_mode + symptom_mode).

---

# ✅ How To Run (Command Order)

Below are complete **step-by-step execution pipelines** for each task.

**Important: Commands are examples; paths and hyperparameters can be changed.**

---

# 🔵 A) Binary Depression Classification (Healthy vs Depressed)

## Step A1 — Preprocess Body

```bash
python symptom_classification/prepare_body_data.py \
  --pose_mode coping \
  --label_keywords CR CRADK SHAM \
  --output_dir ./symptom_classification/processed_body_binary_coping
```

## Step A2 — Preprocess Face 

```bash
python symptom_classification/prepare_face_data.py \
  --pose_mode coping \
  --pattern "CR|CRADK|ADK|SHAM" \
  --output_dir ./symptom_classification/processed_face_binary_coping
```

## Step A3 — Train

### MSN (Body-only)

```bash
python symptom_classification/MSN/msn_train.py \
  --data_dir ./symptom_classification/processed_body_binary_coping \
  --save_path ./symptom_classification/MSN/msn_binary_coping.pth \
  --batch_size 16 \
  --lr 0.001 \
  --epochs 40
```

### Res_Res (Body-only)

```bash
python symptom_classification/Res_Res/resres_train.py \
  --input_type B \
  --body_dir ./symptom_classification/processed_body_binary_coping \
  --epochs 40 \
  --batch_size 16 \
  --lr 0.001 \
  --save_path ./symptom_classification/Res_Res/resres_body_binary_coping.pth
```

### Res_Res (Face-only)

```bash
python symptom_classification/Res_Res/resres_train.py \
  --input_type F \
  --face_dir ./symptom_classification/processed_face_binary_coping \
  --epochs 40 \
  --batch_size 16 \
  --lr 0.0005 \
  --save_path ./symptom_classification/Res_Res/resres_face_binary_coping.pth
```

### Res_Res (Fusion)

```bash
python symptom_classification/Res_Res/resres_train.py \
  --input_type F+B \
  --body_dir ./symptom_classification/processed_body_binary_coping \
  --face_dir ./symptom_classification/processed_face_binary_coping \
  --fusion avg_logit \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.0003 \
  --pretrained \
  --save_path ./symptom_classification/Res_Res/resres_fusion_binary_coping.pth
```

### TSFFM (Fusion)

```bash
python symptom_classification/TSFFM/tsffm_train.py \
  --body_dir ./symptom_classification/processed_body_binary_coping \
  --face_dir ./symptom_classification/processed_face_binary_coping \
  --epochs 40 \
  --batch_size 8 \
  --lr 0.0001 \
  --pretrained \
  --save_path ./symptom_classification/TSFFM/tsffm_binary_coping.pth
```

## Step A4 — Evaluate

```bash
python symptom_classification/MSN/msn_eval.py \
  --model_path ./symptom_classification/MSN/msn_binary_coping.pth
```

```bash
python symptom_classification/Res_Res/resres_eval.py \
  --input_type F+B \
  --model_path ./symptom_classification/Res_Res/resres_fusion_binary_coping.pth
```

```bash
python symptom_classification/TSFFM/tsffm_eval.py \
  --model_path ./symptom_classification/TSFFM/tsffm_binary_coping.pth
```

---

# 🟡 B) Symptom Classification (Retardation / Agitation)

## Step B1 — Generate Symptom Labels

```bash
python symptom_classification/get_retardation_agitation_labels.py
```

Generates:
- `symptom_classification/labels/retardation_labels.csv`
- `symptom_classification/labels/agitation_labels.csv`

## Step B2 — Preprocess Body

```bash
python symptom_classification/prepare_body_data_symptom_modes.py \
  --symptom_mode agitation \
  --pose_mode coping \
  --output_dir ./symptom_classification/processed_body_agitation_coping
```

## Step B3 — Preprocess Face 

```bash
python symptom_classification/prepare_face_data_symptom_modes.py \
  --symptom_mode agitation \
  --pose_mode coping \
  --output_dir ./symptom_classification/processed_face_agitation_coping
```

## Step B4 — Train (examples)

### MSN (Body-only)

```bash
python symptom_classification/MSN/msn_train.py \
  --data_dir ./symptom_classification/processed_body_agitation_coping \
  --save_path ./symptom_classification/MSN/msn_agitation_coping.pth \
  --batch_size 16 \
  --lr 0.001 \
  --epochs 40
```

### Res_Res (Fusion)

```bash
python symptom_classification/Res_Res/resres_train.py \
  --input_type F+B \
  --body_dir ./symptom_classification/processed_body_agitation_coping \
  --face_dir ./symptom_classification/processed_face_agitation_coping \
  --fusion avg_logit \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.0003 \
  --save_path ./symptom_classification/Res_Res/resres_fusion_agitation_coping.pth
```

### TSFFM (Fusion)

```bash
python symptom_classification/TSFFM/tsffm_train.py \
  --body_dir ./symptom_classification/processed_body_agitation_coping \
  --face_dir ./symptom_classification/processed_face_agitation_coping \
  --epochs 40 \
  --batch_size 8 \
  --lr 0.0001 \
  --save_path ./symptom_classification/TSFFM/tsffm_agitation_coping.pth
```

## Step B5 — Evaluate

```bash
python symptom_classification/MSN/msn_eval.py \
  --model_path ./symptom_classification/MSN/msn_agitation_coping.pth
```

---

# 🔴 C) Rating Trend Classification

## Step C1 — Compute Rating Differences

```bash
python rating_trend/calculate_rating_diffs.py
```

Generates:
- `rating_trend/rating_differences.csv`

## Step C2 — Preprocess Body Trials

```bash
python rating_trend/prepare_body_trends.py \
  --output_dir ./rating_trend/processed_body_trends_exp1
```

## Step C3 — Preprocess Face Trials (optional)

```bash
python rating_trend/prepare_face_trends.py \
  --output_dir ./rating_trend/processed_face_trends_exp1
```

## Step C4 — Train

### MSN (Body-only, Multi-task: predicts BOTH `label_n` and `label_p`)

```bash
python rating_trend/MSN/msn_train.py \
  --data_dir ./rating_trend/processed_body_trends_exp1 \
  --epochs 30 \
  --batch 16 \
  --lr 3e-4 \
  --save ./rating_trend/MSN/msn_trend_exp1.pth
```

✅ No `--label` argument is needed.

### Res_Res (Single target; choose `label_n` or `label_p`)

```bash
python rating_trend/Res_Res/resres_train.py \
  --input_type F+B \
  --body_dir ./rating_trend/processed_body_trends_exp1 \
  --face_dir ./rating_trend/processed_face_trends_exp1 \
  --label label_n \
  --fusion avg_logit \
  --epochs 60 \
  --batch_size 16 \
  --lr 3e-4 \
  --save_path ./rating_trend/Res_Res/resres_trend_label_n_exp1.pth
```

### TSFFM (Single target; choose `label_n` or `label_p`)

```bash
python rating_trend/TSFFM/tsffm_train.py \
  --body_dir ./rating_trend/processed_body_trends_exp1 \
  --face_dir ./rating_trend/processed_face_trends_exp1 \
  --label label_p \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-4 \
  --save_path ./rating_trend/TSFFM/tsffm_trend_label_p_exp1.pth
```

## Step C5 — Evaluate

```bash
python rating_trend/MSN/msn_eval.py \
  --model_path ./rating_trend/MSN/msn_trend_exp1.pth
```

---

# 🧩 Model Summary

| Model   | Binary/Symptom | Rating Trend | Target Selection |
|---------|-----------------|--------------|------------------|
| MSN     | Single-task     | Multi-task   | No (trend predicts both) |
| Res_Res | Single-task     | Single-task  | Yes (`label_n` or `label_p`) |
| TSFFM   | Single-task     | Single-task  | Yes (`label_n` or `label_p`) |

---

# 📝 Notes

- Always keep preprocessing output directories consistent with training inputs.
- For reproducibility, use separate `processed_*` folders for each configuration.