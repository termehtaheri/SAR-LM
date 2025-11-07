# 🎧 SAR-LM: Symbolic Audio Reasoning with Large Language Models

**Authors:** Termeh Taheri, Yinghao Ma, and Emmanouil Benetos  
**Affiliation:** Centre for Digital Music (C4DM), Queen Mary University of London  
**Paper:** _SAR-LM: Symbolic Audio Reasoning with Large Language Models_ (to appear on arXiv, 2025)

---

## Overview

**SAR-LM** is a modular framework for **symbolic audio reasoning** — combining feature extraction, captioning, and reasoning within a single, transparent pipeline.

Instead of treating audio as raw waveforms only, SAR-LM converts it into **symbolic representations** (speech transcripts, event tags, note sequences, chord progressions, etc.) and feeds them into large language models such as **Gemini**, **Qwen-3**, and **Qwen-Omni** for reasoning over sound.

This design enables interpretability, reproducibility, and controlled evaluation on reasoning benchmarks such as **MMAU**, **MMAR**, and **OmniBench**.

---

## ✨ Key Features

- 🔊 **Unified Extractors** – PANNs, Whisper, MT3, Musicnn, Chordino, and DAWN emotion features.  
- 🧠 **Multi-Backend Reasoning** – Gemini 2.5 Pro, Qwen-3, and Qwen-Omni backends for symbolic QA.  
- 🗣️ **Captioning Pipelines** – Symbolic, Mixed, and End-to-End audio caption generation.  
- 🧩 **Fully Modular Design** – Each extractor and reasoner is containerized and can run independently.  
- 📊 **Reproducible Outputs** – JSON-based I/O for easy integration with benchmarks and analysis tools.

---

## 🧱 Repository Structure

```
SAR-LM/
│
├── src/sar_lm/
│   ├── extractors/         # Individual feature extractors (PANNs, Whisper, etc.)
│   ├── captions/           # Symbolic, mixed, and end-to-end captioners
│   ├── reasoners/          # Gemini, Qwen3, Qwen-Omni reasoning backends
│   ├── prompts/            # All prompt templates (centralized)
│   └── pipelines/          # Orchestrators for extraction, merging, captioning, reasoning
│
├── examples/               # Sample audios and QA examples
├── outputs/                # Example outputs (features, captions, reasoning results)
├── docker/                 # Dockerfiles for all modules
├── requirements/           # Environment-specific dependencies
├── Makefile                # Workflow shortcuts
├── CITATION.cff            # Citation metadata
├── pyproject.toml          # Package and dependency configuration
└── README.md
```

---

## ⚙️ Installation

### Option 1: Local (Recommended for testing)
```bash
git clone https://github.com/termehtaheri/SAR-LM.git
cd SAR-LM
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/base.txt
```

### Option 2: Docker
Each extractor and reasoning module has its own `Dockerfile` under `docker/`.  
You can build them individually:
```bash
docker-compose build panns
docker-compose build whisper
```

or run all at once:
```bash
docker-compose up -d
```
(All services will start in idle mode and can be triggered independently.)

---

## 🚀 Usage

### 1. Extract features
```bash
PYTHONPATH=src python -m sar_lm.pipelines.extract_pipeline \
  --audio_dir examples \
  --output_dir outputs/features_panns \
  --device cpu
```

### 2. Merge features
```bash
PYTHONPATH=src python -m sar_lm.pipelines.merge_features \
  --panns outputs/features_panns/panns_features.json \
  --whisper outputs/features_whisper/whisper_features.json \
  --mt3 outputs/features_mt3/mt3_features.json \
  --emotion outputs/features_dawn/dawn_emotion_features.json \
  --musicnn outputs/features_musicnn/musicnn_features.json \
  --chordino outputs/features_chordino/chordino_features.json \
  --output outputs/features_merged/features_merged.json
```

### 3. Generate captions
Symbolic captioning:
```bash
PYTHONPATH=src python -m sar_lm.pipelines.captioning_pipeline \
  --mode symbolic \
  --audio_dir examples \
  --features outputs/features_merged/features_merged.json \
  --output outputs/captions/symbolic_captions.json
```

### 4. Run reasoning
```bash
PYTHONPATH=src python -m sar_lm.pipelines.reasoning_pipeline \
  --reasoner qwen3 \
  --features outputs/features_merged/features_merged.json \
  --qa examples/sample_qa.json \
  --output outputs/reasoning/qwen3_results.json
```

---

## 🔐 API Keys

If you use **Gemini** models for captioning or reasoning, set your API key in a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

---

## 🧩 Reproducibility

All environments are defined in:
- `requirements/*.txt` – for lightweight installs  
- `docker/extractors/` – containerized extractors  
- `requirements/mt3_env.yml` – specialized MT3 setup  

To build everything cleanly:
```bash
make env
```

---

## 📚 Citation

If you use SAR-LM in your work, please cite:

```
@article{taheri2025sarlm,
  title={SAR-LM: Symbolic Audio Reasoning with Large Language Models},
  author={Taheri, Termeh and Ma, Yinghao and Benetos, Emmanouil},
  journal={arXiv preprint arXiv:TBD},
  year={2025}
}
```

---

## 🧠 Acknowledgements

This project was developed at the **Centre for Digital Music (C4DM)**,  
**Queen Mary University of London**, as part of Termeh Taheri’s MSc research project supervised by Prof. Emmanouil Benetos.  

Special thanks to Yinghao Ma for guidance on benchmarking and integration.

---

## 🪪 License

This repository is released under the **MIT License**.  
See [LICENSE](LICENSE) for details.
