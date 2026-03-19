# DINO-WM Course Project Plan

## Goal

Keep the original DINO-WM pipeline fixed:

`image encoder -> latent transition model -> planning`

Then study two axes:

1. Representation study:
   compare `DINOv2 patch` and `DINOv2 CLS`
2. Dynamics study:
   compare `deterministic` and `gaussian` latent transitions

## Current Scope

Implemented:

- `DINOv2 patch + deterministic`
- `DINOv2 CLS + deterministic`
- `DINOv2 patch + gaussian`
- `DINOv2 CLS + gaussian`

Not yet implemented:

- `DINOv3`
- `V-JEPA`
- `DINO-Tok`
- `VFM-VAE`

This is the right scope for the course project because it answers both required ablation questions without adding high-risk model integration work.

## Environment

Execution environment used in practice:

- WSL2 `Ubuntu-24.04`
- `micromamba` environment `dino_wm`
- Python `3.9.19`
- MuJoCo `2.1`
- CUDA-enabled PyTorch

Important practical setting:

- use `env.num_workers=0` when running from `/mnt/c/...`

## Completed Work

### 1. Sanity-stage implementation

Completed:

- pretrained PointMaze planning baseline
- deterministic patch training
- deterministic CLS training
- Gaussian patch training
- Gaussian CLS training
- planning with all four resulting checkpoints

### 2. Formal matrix

Completed:

- three seeds: `0, 1, 2`
- four combinations:
  - patch deterministic
  - CLS deterministic
  - patch gaussian
  - CLS gaussian
- training setup:
  - `epochs=3`
  - `n_rollout=64`
  - `n_evals=3`

This produced a full `3 x 4` formal matrix.

### 3. Larger-sample planning follow-up

Reason:

- `n_evals=3` made `success_rate` saturate

What was done:

- planning visualization disabled for larger runs
- video saving disabled
- `eval_seed` generation fixed
- direct `n_evals=10` tested
- stable fallback introduced:
  two shards with `n_evals=5`, using base seeds `0` and `5`

Currently completed larger-sample runs:

- seed-0 deterministic patch
- seed-0 deterministic CLS, shard A
- seed-0 deterministic CLS, shard B
- seed-0 Gaussian patch, shard A
- seed-0 Gaussian patch, shard B

## Current Findings

From the three-seed formal matrix:

- `patch + deterministic` has the best average `mean_state_dist`
- `CLS + deterministic` is very close
- Gaussian models achieve better likelihood-style training losses
- Gaussian models do not currently improve planning quality

From the larger-sample seed-0 follow-up:

- deterministic patch remains stronger than patch Gaussian
- deterministic CLS remains competitive with deterministic patch
- `mean_state_dist` is more informative than `success_rate`

## Remaining Tasks

### Must do

- complete larger-sample planning for `seed0 CLS gaussian`
- complete larger-sample planning for seeds `1` and `2`
- write the larger-sample results back into all docs
- finalize the report wording around what is and is not supported by the data

### Optional

- integrate one third representation:
  `DINOv3` or `V-JEPA`

Only do this if time remains after the larger-sample planning table is complete.

## Recommended Next Order

1. `seed0 CLS gaussian`, shard A and shard B
2. `seed1 patch deterministic`, shard A and shard B
3. `seed1 CLS deterministic`, shard A and shard B
4. `seed1 patch gaussian`, shard A and shard B
5. `seed1 CLS gaussian`, shard A and shard B
6. repeat the same pattern for `seed2`
7. aggregate the larger-sample metrics
8. update report tables and conclusion

## Deliverable Standard

The project is already beyond the proposal stage. A defensible final submission should include:

- one complete implementation section
- one complete three-seed formal matrix
- one partially or fully completed larger-sample planning follow-up
- a careful conclusion that avoids overclaiming from saturated `success_rate`

## Progress Snapshot

Completed:

- core implementation for the course-project scope
- full three-seed formal matrix
- partial larger-sample planning reevaluation on seed 0

In progress:

- larger-sample reevaluation for the remaining configurations

Pending:

- larger-sample seed-0 `CLS + gaussian`
- larger-sample seed-1 matrix
- larger-sample seed-2 matrix
- any third-representation extension

Estimated remaining time for the current reevaluation-only plan:

- about `5` to `8` hours
