# FishMCA

FishMCA is a monocular multi-cue framework for multiple fish tracking in aquaculture videos.

This public repository provides the main code framework, model structure, and qualitative tracking examples. Detailed training settings, private datasets, annotations, pretrained weights, and unpublished research notes are not included before the paper is formally released.

## Overview

FishMCA follows a detector-tracker pipeline:

```text
Input frame
  -> Backbone + Encoder
  -> Detection / Depth / ReID branches
  -> ByteTrack-style online association
  -> Fish trajectories
```

The model predicts bounding boxes, confidence scores, relative depth cues, and appearance embeddings from monocular video frames. These cues are used for online identity association.

## Repository

```text
configs/          Configuration templates
src/              Main FishMCA implementation
ByteTrack/        Tracking components
tools/            Utility scripts
assets/results/   Qualitative tracking videos
train.py          Training entry
infer.py          Inference entry
eval.py           Evaluation entry
```

## Data

The project is designed for self-collected fish videos in MOT-style format. Public datasets are not required.

Private videos, annotations, masks, pseudo labels, and model weights are not included.

## Usage

Create the environment:

```bash
conda env create -f environment.yml
conda activate deptr
```

Example inference entry:

```bash
python infer.py \
  -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml \
  -r path/to/checkpoint.pth
```

Update local paths in the configuration files before running.

## Qualitative Results

Example tracking videos are provided in:

```text
assets/results/Sequence1_tracked.mp4
assets/results/Sequence2_tracked.mp4
assets/results/Sequence3_tracked.mp4
assets/results/Ras_tracked.mp4
```

## Release Note

This repository is a public code preview. Full training details, pretrained weights, and complete experimental settings will be released after the manuscript is formally published.

## Acknowledgements

This project builds on components and ideas from D-FINE, ByteTrack, SAM2, Video Depth Anything, FastReID, DepTR, and PSTR.

## Citation

Citation information will be updated after publication.
