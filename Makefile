# Basic workflow shortcuts

# create environment
env:
	python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt

# extract features
extract:
	PYTHONPATH=src python -m sar_lm.pipelines.extract_pipeline --audio_dir examples --output_dir outputs/features_panns --device cpu

# merge features
merge:
	PYTHONPATH=src python -m sar_lm.pipelines.merge_features --panns outputs/features_panns/panns_features.json --whisper outputs/features_whisper/whisper_features.json --mt3 outputs/features_mt3/mt3_features.json --emotion outputs/features_dawn/dawn_emotion_features.json --musicnn outputs/features_musicnn/musicnn_features.json --chordino outputs/features_chordino/chordino_features.json --output outputs/features_merged/features_merged.json

# reasoning
reason:
	PYTHONPATH=src python -m sar_lm.pipelines.reasoning_pipeline --reasoner qwen3 --features outputs/features_merged/features_merged.json --qa examples/sample_qa.json --output outputs/reasoning/qwen3_results.json

# captioning
caption:
	PYTHONPATH=src python -m sar_lm.pipelines.captioning_pipeline --mode symbolic --audio_dir examples --features outputs/features_merged/features_merged.json --output outputs/captions/symbolic_captions.json
