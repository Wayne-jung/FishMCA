# FishMCA

FishMCA is a monocular multi-cue framework for multiple fish tracking in aquaculture videos. The project is built around a detector-tracker pipeline that predicts detection boxes, confidence scores, instance-level depth cues, and appearance embeddings from a single input frame, then combines spatial, depth, and appearance distances for online identity association.

The code is intended for experiments on self-collected MOT-format fish videos.

> Note: the corresponding manuscript is not formally published yet. Some research notes, manuscript files, data, pretrained weights, and implementation details are intentionally excluded from public release.

## Highlights

- Multiple fish tracking from top-view monocular videos.
- MOT-format video annotation support.
- Detector-side prediction of box, score, relative depth, and appearance embedding.
- Teacher-side supervision during training using SAM2 masks and monocular depth estimation.
- Mask-guided ReID teacher supervision based on FastReID-style training.
- ByteTrack-based online association with spatial, depth, and appearance cues.
- No extra segmentation, depth, or teacher network is required during inference.

## Framework Overview

FishMCA follows a teacher-student design during training.

The student model contains three parallel prediction branches after the shared backbone and encoder:

```text
Input frame
  -> Backbone + Hybrid Encoder
  -> Shared multi-scale visual memory
       |-> Detection Decoder -> boxes and confidence scores
       |-> Depth Decoder     -> instance-level relative depth
       |-> ReID Decoder      -> L2-normalized appearance embeddings
       |
       v
  ByteTrack-style online association
```

The depth decoder and ReID decoder are guided by the detection decoder outputs, so each predicted instance has aligned spatial, depth, and appearance cues.

During training:

1. SAM2 generates instance masks from annotated fish boxes.
2. A monocular depth model provides depth supervision for each fish instance.
3. A FastReID-style Mask ReID Teacher provides clean appearance targets from mask-purified fish crops.
4. The student detector learns detection, relative depth, and appearance representations jointly.

During inference:

1. Only the student detector is used.
2. The detector outputs bounding boxes, scores, depth values, and ReID embeddings.
3. ByteTrack associates detections with existing tracks using a fused multi-cue cost.

The teacher-side models are used only to generate training supervision and are removed at inference time.

## Repository Structure

```text
configs/                 Training and model configuration files
src/                     Main FishMCA / D-FINE implementation
ByteTrack/               Tracking backend
assets/results/          Qualitative tracking result videos
tools/                   Export, benchmark, and utility scripts
train.py                 Main training entry
infer.py                 Inference entry
eval.py                  Evaluation entry
evaluate_track_with_depth.py
                         Tracking evaluation helper
```

## Data Format

This project assumes self-collected fish videos annotated in MOT format.

A typical sequence is expected to follow a structure similar to:

```text
YourDataset/
  train/
    Sequence001/
      img1/
        000001.jpg
        000002.jpg
        ...
      gt/
        gt.txt
  test/
    Sequence002/
      img1/
      gt/
```

The identity labels in `gt.txt` are used for ReID supervision. If different videos reuse the same local track IDs, the loader maps `(video_name, track_id)` to a global identity internally.

If your current training pipeline converts MOT annotations into COCO-style JSON, that is only an internal annotation format for detector training. It does not mean that COCO or any other public dataset is required.

## Installation

Create the environment:

```bash
conda env create -f environment.yml
conda activate deptr
```

Install SAM2:

```bash
git clone https://github.com/facebookresearch/sam2.git SAM2
cd SAM2
pip install -e .
cd ..
```

Install or place the monocular depth dependency according to your local setup. The original experiments use a VideoDepthAnything-style teacher for training-time depth supervision.

Download the required teacher-side checkpoints locally:

```bash
# SAM2 checkpoint
mkdir -p SAM2/checkpoints

# VideoDepthAnything / depth checkpoint
mkdir -p VideoDepthAnything
```

Place pretrained weights in the paths referenced by your configuration files.

## Configuration

The main custom configuration is:

```text
configs/dfine/custom/dfine_hgnetv2_l_custom.yml
```

The shared D-FINE/FishMCA settings are in:

```text
configs/dfine/include/dfine_hgnetv2.yml
```

For online Mask ReID Teacher supervision, set:

```yaml
DFINE:
  reid_teacher_config: path/to/fish_mask_reid_R50.yml
  reid_teacher_checkpoint: path/to/mask_reid_teacher.pth
  reid_teacher_input_size: [256, 128]
```

If these fields are `null`, the model still runs without teacher-side ReID distillation. In that case, only the available detection, depth, and track-ID based losses are used.

## Training

Example training command:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=4444 --nproc_per_node=2 train.py \
  -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml \
  --use-amp \
  --seed=0 \
  --track
```

Before training, update the custom dataset paths in:

```text
configs/dataset/custom_detection.yml
```

## Evaluation

Run evaluation with a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py \
  -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml \
  --test-only \
  -r ./output/dfine_hgnetv2_l_custom/last.pth
```

For tracking evaluation on MOT-format videos, use the project-specific evaluation helper:

```bash
python evaluate_track_with_depth.py
```

Adjust dataset paths, sequence names, and checkpoint paths before running.

## Qualitative Results

Example tracking videos are provided for visual inspection:

```text
assets/results/Sequence1_tracked.mp4
assets/results/Sequence2_tracked.mp4
assets/results/Sequence3_tracked.mp4
assets/results/Ras_tracked.mp4
```

These videos are qualitative demonstrations only. Raw training videos, annotations, masks, pseudo labels, and model weights are not included in the public repository.

## Inference

Example:

```bash
python infer.py \
  -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml \
  -r ./output/dfine_hgnetv2_l_custom/last.pth
```

The inference model does not require SAM2, VideoDepthAnything, or the FastReID teacher to run online. These models are only used for training-time supervision.

## Important Notes

- This repository is designed for self-collected MOT-format fish videos.
- Public datasets are not required.
- Raw videos, annotations, masks, pseudo labels, and weights should not be committed.
- Manuscript files and private research notes are ignored by default.
- If the repository is made public before the paper is published, consider releasing only the minimal reproducibility code and keeping core research modules private.

## Acknowledgements

This project builds on ideas and components from:

- D-FINE
- ByteTrack
- SAM2
- Video Depth Anything
- FastReID
- DepTR

Please cite the corresponding original projects when using their components.

## Citation

The FishMCA manuscript is currently under preparation/review. Citation information will be updated after formal publication.
