# floodkd

Self-supervised cross-modal knowledge distillation for flood segmentation
using Sentinel-1 SAR and Sentinel-2 optical imagery.

Sentinel-2 teaches. Sentinel-1 learns. At inference only Sentinel-1 is needed,
because that is the only sensor that still works under the cloud cover that
accompanies a flood.

---

## Folder structure

```
floodkd/
├── run.py                 ONE entry point. Every experiment starts here.
├── configs/
│   └── default.yaml       Every number lives here. No magic numbers in code.
├── src/
│   ├── config.py          Loads YAML, applies CLI overrides, rejects typos
│   ├── data.py            Downloads chips, PyTorch Dataset, normalisation
│   ├── models.py          U-Net + the TeacherStudent wrapper
│   ├── losses.py          Segmentation losses + the three distillation losses
│   ├── metrics.py         IoU / F1 / precision / recall, ignore mask applied ONCE
│   └── engine.py          Train + eval loops, checkpoints, CSV logging, seeding
├── data/                  Downloaded chips (git-ignored, ~14 GB if you take it all)
└── runs/                  One folder per experiment: config.json, log.csv, best.pt
```

Rule of thumb: **`src/` never prints and never decides anything. `run.py` does.**
That is what keeps the modules reusable when you later write the Chapter 4
figures from a notebook.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## The pipeline

Four stages. Run them in order.

### 1. Look at the data first

```bash
python run.py explore
```

Downloads one chip, prints the label balance, plots S1 / S2 / label side by side.

### 2. Get the data

```bash
python run.py download --split train,val,test
python run.py stats --split train
```

`stats` prints how much of the dataset is water, land, and no-data. You need
those numbers to justify the class weights in the config.

### 3. Baselines — you cannot claim an improvement without these

```bash
python run.py train --modality s1 --out-dir runs/base_s1
python run.py train --modality s2 --out-dir runs/teacher_s2
```

The first is the SAR-only baseline your framework must beat. The second is
both the optical baseline **and** the teacher for stage 4. Together they
reproduce the SAR-vs-optical gap that STURM-Flood reports, on your own data.

### 4. The proposed framework

```bash
python run.py distill --teacher-ckpt runs/teacher_s2/best.pt --out-dir runs/distill
```

Trains the SAR student with the frozen optical teacher. Validation is on
**Sentinel-1 only**, so the number you report is an honest deployment number.

### 5. Test once, at the end

```bash
python run.py eval --ckpt runs/distill/best.pt --split test --modality s1
python run.py eval --ckpt runs/distill/best.pt --split bolivia --modality s1
```

Bolivia is a separate held-out event. Never train or tune on it — it is your
generalisation evidence, and it answers the criticism Portalés-Julià et al.
(2025) make of single-dataset evaluation.

---

## Ablations

The point of a config file is that the ablation table writes itself:

| Run | Command |
|---|---|
| No distillation (baseline) | `train --modality s1` |
| Response KD only | `distill --alpha 1 --beta 0` |
| Pair-wise KD only | `distill --alpha 0 --beta 1` |
| Both | `distill --alpha 1 --beta 1` |
| Both + confidence gating | `distill --alpha 1 --beta 1 --gate-threshold 0.7` |

Each writes its own `runs/<name>/log.csv` and `config.json`, so months later you
can still prove which settings produced which row of your results table.

---

## Design decisions worth defending in a viva

**The ignore mask exists in one place.** Labels are `-1` (no data), `0` (land),
`1` (water). If those `-1` pixels leak into the loss or the metric, every number
you report is quietly wrong. `metrics.py` masks them once, `losses.py` passes
`ignore_index=-1`, and nothing else is allowed to touch it. In a small test,
forgetting this turns a true IoU of 0.40 into 0.25 — wrong, and with no error
message.

**Official splits, not random ones.** `data.py` reads the split CSVs published
with the dataset. Random splits would put chips from the same flood event in
both train and test, and inflate the results.

**The teacher is frozen.** `TeacherStudent` disables gradients on the teacher.
Only the SAR student learns, so any improvement is attributable to the transfer.

**Inference uses one modality.** `TeacherStudent.predict()` takes `s1` only.
The architecture makes the deployment claim structurally true rather than a
promise in the text.

**Seeds are fixed and configs are saved.** `set_seed()` covers Python, NumPy and
torch. Every run writes `config.json` next to its checkpoint.

**Checkpoints: `best.pt` and `last.pt`.** Best is selected on validation water
IoU, not on loss.

---

## What is finished and what is not

Done and correct: the data pipeline, normalisation, metrics, segmentation and
distillation losses, U-Net, training loops, checkpointing, logging, CLI.

Not done yet, and deliberately so:

- **Self-supervised pre-training of the teacher.** Right now the teacher is
  trained supervised on Sentinel-2. That is the honest baseline. Replacing it
  with a self-supervised teacher is the actual novelty of your project, and it
  belongs in Chapter 3 once you have chosen a pretext task.
- **Data augmentation.** The `transform` hook exists in `Sen1Floods11` but is
  unused. Add flips and rotations once the baseline runs.
- **The weakly-labelled split.** Only the 446 hand-labelled chips are wired up.
  The much larger weak set is what self-supervised pre-training will need.

---

## Suggested first session

```bash
python run.py explore
python run.py download --split val --modality s1
python run.py train --modality s1 --split val --epochs 2 --batch-size 2 --device cpu
```

Two epochs on the small validation split, on CPU. The IoU will be poor — that is
fine. The purpose is to prove the whole loop runs end to end before you commit
to a long training job.
