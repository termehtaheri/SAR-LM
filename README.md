<h1 align="center">SAR-LM: Symbolic Audio Reasoning with Large Language Models</h1>

<p align="center">
  <img src="./static/images/pipeline.gif" width="85%" alt="SAR-LM Pipeline Overview" style="border-radius: 12px;">
</p>

<p align="center">
  <a href="https://github.com/termehtaheri/SAR-LM">
    <img src="https://img.shields.io/badge/GitHub-black.svg" alt="GitHub Repo">
  </a>
  <a href="https://arxiv.org/abs/2511.06483">
    <img src="https://img.shields.io/badge/arXiv-2511.06483-red.svg" alt="arXiv Paper">
  </a>
</p>

---

## 🎧 Overview

**SAR-LM (Symbolic Audio Reasoning with Large Language Models)** is a modular framework designed to enable interpretable reasoning over audio.  
Instead of relying on dense, opaque embeddings, SAR-LM converts audio into symbolic, human-readable representations including speech, sound events, and music, that can be reasoned over by large language models (LLMs).


SAR-LM achieves competitive accuracy across three major audio reasoning benchmarks (MMAU, MMAR) while maintaining transparent interpretability, allowing clear traceability of model failures to specific features.

---

## 🧩 Method and Novelty

SAR-LM introduces a symbolic audio reasoning pipeline, built around four modular stages:

1. **Symbolic Feature Extraction** – Uses pretrained and signal-processing models to extract transcripts (Whisper), sound events (PANNs), emotions (DAWN), chords (Chordino), musical notes (MT3), and tags (Musicnn).  
2. **Prompt Construction** – Converts symbolic features into structured natural language prompts that pair with benchmark questions.  
3. **Caption Generation** – Summarizes symbolic features into concise natural-language captions for easier reasoning.  
4. **Reasoning with LLMs** – Performs multi-choice reasoning using open- or closed-source LLMs (e.g., Qwen3, Gemini 2.5 Pro).

Unlike dense audio embeddings (e.g., CLAP, BEATs), SAR-LM’s symbolic inputs are explicit and interpretable, enabling detailed error analysis and content-aware reasoning.  
The pipeline supports flat symbolic, symbolic caption, and end-to-end caption reasoning modes.

---

## 📊 Results on MMAU Benchmark

SAR-LM outperforms all prior methods on the MMAU benchmark, demonstrating the effectiveness of symbolic reasoning:

| **Method** | **Sound** | **Music** | **Speech** | **Overall** |
|-------------|------------|------------|-------------|--------------|
| MMAU (Best) | 57.35 | 49.70 | 64.86 | 57.30 |
| Audio-CoT | 62.16 | 55.99 | 56.16 | 58.10 |
| Audio-Reasoner | 60.06 | 64.30 | 60.70 | 61.71 |
| **SAR-LM (Gemini + Symbolic)** | **73.27** | **64.97** | **82.28** | **73.5** |

SAR-LM achieves the highest overall accuracy (73.5%), especially excelling in speech reasoning tasks.

---

## ⚙️ Installation and Setup

### Without Docker

You can install and run each module directly using its requirement file:

1. **Clone the repository**
   ```bash
   git clone https://github.com/termehtaheri/SAR-LM.git
   cd SAR-LM
   ```

2. **Create an environment for each extractor**
   ```bash
   conda create -n sar-lm python=3.10
   conda activate sar-lm
   pip install -r requirements/panns.txt
   pip install -r requirements/whisper.txt
   pip install -r requirements/mt3.txt
   # ...and so on for other extractors
   ```

3. **Run a feature extraction pipeline**
   ```bash
   PYTHONPATH=src python -m sar_lm.pipelines.extract_pipeline \
       --audio_dir examples \
       --output_dir outputs/features_panns \
       --device cpu
   ```

4. **Run reasoning**
   ```bash
   PYTHONPATH=src python -m sar_lm.pipelines.reasoning_pipeline \
       --reasoner qwen3 \
       --features outputs/features_merged/features_merged.json \
       --qa examples/sample_qa.json \
       --output outputs/reasoning/qwen3_results.json
   ```

---

### 🐳 With Docker

SAR-LM provides individual Dockerfiles for each feature extractor and reasoning module, defined in the `docker/` folder.

#### Build all containers
```bash
docker-compose build
```

#### Run specific modules
```bash
docker-compose up panns       # run PANNs extractor
docker-compose up whisper     # run Whisper extractor
docker-compose up reasoning   # run reasoning pipeline
```

Each service will automatically process the example audios and save extracted features in `./outputs/features`.

---

## 📁 Project Structure

```
SAR-LM/
│
├── docker/                    # Dockerfiles for each module
├── examples/                  # Sample audio and QA JSON
├── outputs/                   # Extracted features and reasoning outputs
├── requirements/              # Dependencies per extractor
├── src/sar_lm/
│   ├── agents/                # Feature selection agents (e.g. Gemini)
│   ├── captions/              # Captioners (symbolic, mixed, end-to-end)
│   ├── extractors/            # PANNs, Whisper, MT3, DAWN, Chordino, etc.
│   ├── pipelines/             # High-level pipelines for extraction/reasoning
│   ├── prompts/               # Prompt templates
│   └── reasoners/             # LLM reasoning backends (Gemini, Qwen)
│
├── docker-compose.yml
├── LICENSE
└── README.md
```

SAR-LM’s modular design allows easy substitution of extractors, captioners, or reasoners, enabling flexible experimentation and transparent analysis.

---

## 🧾 Citation

If you find **SAR-LM** useful for your research, please cite:

```bibtex
@article{taheri2025sarlm,
  title={SAR-LM: Symbolic Audio Reasoning with Large Language Models},
  author={Taheri, Termeh and Ma, Yinghao and Benetos, Emmanouil},
  journal={arXiv preprint arXiv:2511.06483},
  year={2025},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2511.06483}
}
```

---

## 📜 License

This project is released under the MIT License.  
© 2025 Termeh Taheri.
