# VoxelMind — project handoff for Claude

This repository is a fork of SegVol (`https://github.com/AbdulnafeAsueid/SegVol`, upstream `BAAI-DCAI/SegVol`) for **VoxelMind**, the ICS 619 Master Project of Abdulnafe Asuied (KFUPM, supervisor Dr. Muzammil Behzad). Read this file fully before writing code. It records decisions made with the supervisor; do not change them without asking the user.

## 1. What the project is

- **Goal:** segment pathological findings in 3D chest CT from free-text radiology report sentences, while keeping SegVol's multi-organ segmentation ability.
- **Method:** keep SegVol's 3D ViT image encoder, prompt encoder and mask decoder **frozen**. Replace its CLIP text pathway with an open-weight LLM (**LLaMA-2-7B**, `meta-llama/Llama-2-7b-hf`, base model, gated on Hugging Face), loaded in 4-bit with **QLoRA/LoRA**. Connect the LLM to SegVol through new trainable **cross-attention adapters** that let the decoder attend to the *full sequence* of LLM tokens. Trainable parts: adapters + LoRA weights only.
- **Novelty (approved by supervisor):** M3D-LaMed (Bai et al., 2024, arXiv:2404.00578) already connects LLaMA-2-7B to SegVol, but through a *single* `[SEG]` token mapped by an MLP, and it fine-tunes SegVol. VoxelMind differs by (a) token-level cross-attention, (b) SegVol frozen, (c) training/evaluation on expert-annotated report findings (ReXGroundingCT) with chain-of-thought supervision.
- **Scope:** chest-CT findings (lung and pleural) described in report language. Not whole-body organ deformation. Testing other organs (e.g. indirectly described organs like "the organ that stores bile") is optional future work, only if time allows.

## 2. Supervisor decisions (2026-10-26)

1. Contribution as described above is enough.
2. Baselines: SegVol required; M3D-LaMed "if possible". The user wants **both from the start**.
3. Editing the proposal directly for the narrowed scope is fine.
4. Multi-organ kept as the base; other organs later if time allows.
5. Title change ("Pixel-Perfect" → e.g. "Voxel-Level") deferred to the final report.
6. Start with **LLaMA-2-7B**. Only if results are poor, consider a newer/stronger LLM.

## 3. SegVol code facts (this repo)

- `network/model.py`
  - `SegVol.forward_decoder(image_embedding, img_shape, text, boxes, points)` is the integration point. Text goes through `self.text_encoder(text)` → `text_embedding` of shape **(B, 768)**, which is passed to **both** `self.prompt_encoder(..., text_embedding=...)` and `self.mask_decoder(..., text_embedding=...)`. Inspect `segment_anything_volumetric/modeling/prompt_encoder.py` and `mask_decoder.py` to see exactly how each uses it before designing the adapter interface.
  - `TextEncoder`: frozen `CLIPTextModel`, template `"A computerized tomography of a {}."`, pooled output (512) → `dim_align` Linear(512→768). The tokenizer call uses `padding=True` but **no truncation**, so report sentences longer than 77 CLIP tokens will fail. For the SegVol baseline on ReXGroundingCT, add `truncation=True, max_length=77` and document it.
  - `feat_shape = roi_size / patch_size`. ViT: patch 4×16×16 on input 32×256×256 → 8×16×16 = 2,048 image tokens, hidden size 768, 12 layers, 12 heads.
  - Losses already exist in `utils/loss.py` (BCE + Dice).
- Inference uses zoom-out-zoom-in (global pass on resized 32×256×256 volume, then sliding windows in the ROI at original resolution). See `inference_demo.py`, `utils/monai_inferers_utils.py`, `data_process/`.
- Preprocessing for any new data must match SegVol's exactly (orientation, spacing, intensity clipping/normalization). Read `data_process/` and `inference_demo.py` and reuse them rather than writing new preprocessing.
- `network/model.py` differs from the GitHub version only by one added first line, `# pyright: reportMissingImports=false` (an editor setting; harmless). The file is otherwise the original upstream code.

### Where the adapters plug in (analysed 2026-10-26, verify against the code before implementing)

SegVol uses the text embedding `t` of shape (B, 768) in **two places**:

1. **As one prompt token.** `segment_anything_volumetric/modeling/prompt_encoder.py`, `forward()`: `sparse_embeddings = torch.cat([sparse_embeddings, text_embedding.unsqueeze(dim=1)], dim=1)`. These sparse tokens go into the mask decoder's two-way transformer, where tokens and the 2,048 image tokens cross-attend to each other.
2. **As a similarity map added to the mask logits.** `segment_anything_volumetric/modeling/mask_decoder.py`, `predict_masks()`: `text_embedding_down = self.txt_align_upscaled_embedding(text_embedding)`, dot product with the upscaled image embedding, added to `masks`.

**Critical:** in `network/model.py`, `forward_decoder()` computes `self.text_encoder(text)` inside `with torch.no_grad():`. Gradients would never reach the LLM LoRA weights or the adapters. The new text path must be computed **outside** `no_grad` (keep `no_grad`/frozen params for SegVol itself).

Planned design (implement in this order):

- **A. Single-token variant (M3D-LaMed style; also the ablation and first sanity check).** LLaMA-2-7B output hidden state of a `[SEG]` token (or the last token) → trainable MLP 4096 → 768 → replaces `t` in both places above. No changes to SegVol modules needed.
- **B. Token-level variant (the proposed method).**
  - Project all LLM token hidden states: (B, N, 4096) → trainable projector → (B, N, 768).
  - Path 1: append all N projected tokens to `sparse_embeddings` instead of one token. This requires a small change to `prompt_encoder.forward()` to accept (B, N, 768). The two-way transformer has **no key-padding mask**, so either use batch size 1 per prompt (SegVol already decodes one prompt at a time) or add a padding mask.
  - Path 2: keep a pooled vector (the `[SEG]` token or mean of valid tokens, projected to 768) for the similarity map in `mask_decoder.predict_masks()`.
  - Optional extra adapter: gated cross-attention where the image tokens (B, 2048, 768) attend to the projected LLM tokens before the decoder, `img = img + tanh(g) * CrossAttn(img, txt)` with `g` initialised to 0, so training starts exactly from SegVol's behaviour.
  - Caution: SegVol's frozen decoder was trained with only a few sparse tokens (points, box, one text token). Feeding N text tokens is new for it, so the projector (and optionally the gated adapter) must learn to produce tokens it can use. Compare B with and without the gated adapter.
- Keep variants A and B in the same codebase behind a config flag so the ablation is a one-line change.

## 4. Data

### ReXGroundingCT (main training and evaluation data)
- Hugging Face `rajpurkarlab/ReXGroundingCT` (gated: accept terms, use an HF access token). About 3.6 GB.
- 3,142 non-contrast chest CTs (from CT-RATE), 8,028 findings, 16,301 annotated entities; ~79% focal, ~21% non-focal; lung and pleural findings only.
- **Training split is partially labeled:** up to 3 representative segmentations per finding. Validation/test splits are exhaustive. The loss must not punish correct predictions on unlabeled lesions of multifocal findings (options to compare: down-weight false-positive voxels for multifocal findings; train on crops around annotated instances).
- Includes a chain-of-thought set generated by GPT-4o: treat it as auxiliary supervision and ablate with/without it.
- **Splits (from the dataset card, checked 2026-10-26):** 2,992 train, 50 public validation, 100 test held **privately** (scored only via the ReXrank leaderboard; masks not available). A separate MICCAI 2026 challenge file `MICCAI_challenge_dataset.json` adds a 200-scan `val` split **with masks** and a 300-scan `test` split (masks withheld). The base `dataset.json` (train/val/test) is unchanged. Dataset card says findings span 14 categories.
- **Plan for this project:**
  - **Test set (final results only):** the 50 public validation scans + the 200 MICCAI challenge validation scans (both exhaustively labeled). First verify none of the 200 challenge scans appear in the 2,992 training scans or the 50 validation scans; drop any overlap and record it.
  - **Validation set (tuning, early stopping):** a fixed, seeded hold-out of ~150 scans from the 2,992 training scans (note: these are partially labeled, so validation Dice is a rough guide, not a final number).
  - **Training:** the remaining training scans.
  - **Optional:** submit predictions for the private 100-scan test to the ReXrank leaderboard for an official score.
  - Save the resulting scan-ID lists under `data/splits/` (e.g. `rexgct_train_ids.txt`, `rexgct_val_ids.txt`, `rexgct_test_ids.txt`) and never tune on the test list.

### TotalSegmentator v3.0.0 (preserve healthy anatomy)
- Zenodo DOI 10.5281/zenodo.22688904, file `Totalsegmentator_dataset_v300.zip`, 37.4 GB, 1,939 scans, 117 structures, CC BY 4.0. Download **only on Rescale**, never on the laptop.
- SegVol was trained on TotalSegmentator v1 (1,203–1,204 scans, 104 classes, 80/20 split). Therefore:
  - `data/splits/totalseg_train_ids.txt` (1,204 IDs, all v1 scans): **training only**.
  - `data/splits/totalseg_test_pool_ids.txt` (735 IDs, s1406–s2140, never seen by SegVol): **evaluation only**, never used for training. Split it once, with a fixed seed, into ~100 validation / ~635 test, stratified by age (pediatric vs adult) using v3's `meta.csv`. Report pediatric scans separately.
  - v1 had 104 classes, v3 has 117 (renamed/added): compare only classes present in both when evaluating SegVol.

### Not used
CT-RATE (no masks; source of ReXGroundingCT), M3D-Seg, MetaStructAtlas (PET/CT).

## 5. Baselines and evaluation

- **SegVol (CLIP text):** organ names on the TotalSegmentator test split; finding sentences on the ReXGroundingCT test split.
- **M3D-LaMed:** public checkpoint (e.g. `GoodBaiBai88/M3D-LaMed-Llama-2-7B`), inference only on the ReXGroundingCT test split. Note in results that it was never trained on report findings. Fine-tuning it is a later decision.
- **Single-token ablation:** a `[SEG]`-token version of VoxelMind (same LLM, data, training) to isolate the effect of token-level cross-attention.
- **Metrics:** Dice (primary), NSD, HD95, gIoU/cIoU, instance hit rate. Report by focal vs non-focal findings and lesion size. Three prompt settings: explicit (organ names), descriptive (report sentences), implicit (small manually checked rewritten set).
- **Statistics:** paired comparisons on the same cases (Wilcoxon signed-rank on per-case Dice) and bootstrap confidence intervals.

## 6. Access status

- LLaMA-2-7B: access to `meta-llama/Llama-2-7b-hf` confirmed on 2026-10-26 ("You have been granted access to this model").
- ReXGroundingCT: gated on Hugging Face; confirm access with the same account/token.

## 7. Environment

- Development: user's Windows laptop, 16 GB RAM, data on a secondary drive. Local tests use only 2–3 scans per dataset (data loader, preprocessing, tensor shapes, a forward pass). Make every path configurable (no hard-coded `D:\...`).
- Training/benchmarking: **Rescale Cloud HPC**. GPU type not yet known: keep batch size, precision, gradient checkpointing and model paths in a config file. Target fits a single 40–80 GB GPU (4-bit LLaMA-2-7B + frozen SegVol).
- Code still calls `.cuda()` directly in several places; keep CPU fallbacks in mind for local debugging.

### Disk space on the laptop (important)

The C: drive is nearly full; D: has ~500 GB free. Keep everything on D:.
- The user sets `HF_HOME=D:\cache\huggingface`, `PIP_CACHE_DIR=D:\cache\pip`, `TORCH_HOME=D:\cache\torch`, and conda `envs_dirs`/`pkgs_dirs` on D:. Check these are in effect (`echo $env:HF_HOME`) before installing or downloading anything; if not, stop and tell the user.
- Create the Python environment on D: (e.g. `conda create --prefix D:\conda\envs\voxelmind python=3.10`).
- Never download LLaMA-2-7B, M3D-LaMed or full datasets on the laptop. For local tests use a tiny stand-in causal LM (e.g. a few-million-parameter test model from Hugging Face) and 2–3 sample scans; the real models and data are downloaded only on Rescale.
- Write outputs (checkpoints, logs, predictions) under a configurable directory on D:.

## 8. Build order

1. Environment file (requirements/conda) and config system.
2. Data loaders for ReXGroundingCT and TotalSegmentator using SegVol's preprocessing and the split files; a test-pool split script (fixed seed, age-stratified).
3. Evaluation module (all metrics above) with unit tests on synthetic masks.
4. Baseline runners: SegVol (with CLIP truncation fix) and M3D-LaMed. Their results form report **R3**.
5. VoxelMind model: LLM wrapper (4-bit + LoRA), then variant A (single token), then variant B (token-level, see section 3). Move the text path out of `no_grad`.
6. Training script (mixed batches of ReXGroundingCT and TotalSegmentator, BCE + Dice), checkpointing, logging.
7. Rescale job scripts (download data with HF token, run training/eval).

Test each step locally on the small sample before moving to Rescale.

## 9. Course milestones (ICS 619) — background only

Listed only so you understand where the code fits. **Do not write report text unless the user asks.** The current focus is the code. Milestones: R1 background and related work (done, Section 7 of the proposal document); R2 technical analysis of SegVol, M3D-LaMed, SAM-Med3D, SAT, LISA; R3 experiments with existing solutions (the baselines); R4 proposed-method experiments; R5 final report. When results are produced, save them (tables as CSV, configs, logs) so they can be used in R3/R4 later.

## 10. Working style

The user is a master's student. Explain non-obvious choices briefly, keep changes small and committed step by step, and never claim something works on real data unless it was actually run.
