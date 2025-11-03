# src/sar_lm/extractors/mt3.py
"""
MT3Extractor
------------
Symbolic music transcription and feature extraction using the MT3 model.
This module converts audio recordings into MIDI files and subsequently
extracts symbolic note-level features (pitch, instrument, onset, offset)
for use in multimodal reasoning tasks.
"""

import os
import json
import librosa
import numpy as np
import note_seq
import pretty_midi
import functools
import gin
import nest_asyncio
import tensorflow.compat.v2 as tf
import jax
import t5x
import seqio
import t5
from tqdm import tqdm
from typing import Dict, Any, List
from mt3 import spectrograms, vocabularies, preprocessors, note_sequences, metrics_utils, models, network
from pretty_midi import program_to_instrument_name, note_number_to_name
from .base import ExtractorBase

nest_asyncio.apply()

SAMPLE_RATE = 16000


# -------------------------------------------------------------------------
# STAGE 1: Audio → MIDI (Transcription)
# -------------------------------------------------------------------------
class MT3Transcriber:
    """Wrapper for the MT3 (T5X) model for automatic music transcription."""

    def __init__(self, checkpoint_dir: str, gin_dir: str, model_type: str = "mt3"):
        """Initialize MT3 inference model.

        Args:
            checkpoint_dir (str): Path to pretrained checkpoint directory.
            gin_dir (str): Directory containing model .gin configuration files.
            model_type (str): Either 'mt3' or 'ismir2021'.
        """
        self.model_type = model_type
        self.batch_size = 8
        self.sample_rate = SAMPLE_RATE

        # Model setup
        self._configure_model(model_type, gin_dir)
        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)
        self.model = self._load_model()
        self._restore_from_checkpoint(checkpoint_dir)

    def _configure_model(self, model_type: str, gin_dir: str):
        """Configure spectrograms, vocabularies, and gin bindings."""
        if model_type == "ismir2021":
            num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
        elif model_type == "mt3":
            num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
        else:
            raise ValueError("Unknown model_type")

        self.outputs_length = 1024
        self.sequence_length = {"inputs": self.inputs_length, "targets": self.outputs_length}

        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins)
        )
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }

        gin_files = [os.path.join(gin_dir, "model.gin"), os.path.join(gin_dir, f"{model_type}.gin")]
        gin_bindings = [
            "from __gin__ import dynamic_registration",
            "from mt3 import vocabularies",
            "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
            "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS",
        ]
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(gin_files, gin_bindings, finalize_config=False)

    def _load_model(self):
        model_config = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_config)
        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features["inputs"].vocabulary,
            output_vocabulary=self.output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config),
        )

    def _restore_from_checkpoint(self, checkpoint_path: str):
        train_state_initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def,
            init_fn=self.model.get_initial_variables,
            input_shapes=self.input_shapes,
            partitioner=self.partitioner,
        )
        restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
            path=checkpoint_path, mode="specific", dtype="float32"
        )
        train_state_axes = train_state_initializer.train_state_axes
        self._predict_fn = self._get_predict_fn(train_state_axes)
        self._train_state = train_state_initializer.from_checkpoint_or_scratch(
            [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0)
        )

    @property
    def input_shapes(self):
        return {
            "encoder_input_tokens": (self.batch_size, self.inputs_length),
            "decoder_input_tokens": (self.batch_size, self.outputs_length),
        }

    def _get_predict_fn(self, train_state_axes):
        def partial_predict_fn(params, batch, decode_rng):
            return self.model.predict_batch_with_aux(params, batch, decoder_params={"decode_rng": None})

        return self.partitioner.partition(
            partial_predict_fn,
            in_axis_resources=(train_state_axes.params, t5x.partitioning.PartitionSpec("data",), None),
            out_axis_resources=t5x.partitioning.PartitionSpec("data",),
        )

    def _audio_to_dataset(self, audio: np.ndarray):
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors({"inputs": frames, "input_times": frame_times})

    def _audio_to_frames(self, audio: np.ndarray):
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode="constant")
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    def preprocess(self, ds):
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key="inputs",
                additional_feature_keys=["input_times"],
            ),
            preprocessors.add_dummy_targets,
            functools.partial(preprocessors.compute_spectrograms, spectrogram_config=self.spectrogram_config),
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    def postprocess(self, tokens, example):
        tokens = np.array(tokens, np.int32)
        if vocabularies.DECODED_EOS_ID in tokens:
            tokens = tokens[: np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
        start_time = example["input_times"][0]
        start_time -= start_time % (1 / self.codec.steps_per_second)
        return {"est_tokens": tokens, "start_time": start_time, "raw_inputs": []}

    def __call__(self, audio: np.ndarray):
        """Run full MT3 inference on a waveform and return a NoteSequence."""
        ds = self._audio_to_dataset(audio)
        ds = self.preprocess(ds)
        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
            ds, task_feature_lengths=self.sequence_length
        ).batch(self.batch_size)

        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), self._predict(model_ds)):
            predictions.append(self.postprocess(tokens, example))

        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec
        )
        return result["est_ns"]

    def _predict(self, model_ds):
        for batch in model_ds.as_numpy_iterator():
            prediction, _ = self._predict_fn(self._train_state.params, batch, jax.random.PRNGKey(0))
            yield from self.vocabulary.decode_tf(prediction).numpy()

    def transcribe_file(self, audio_path: str, output_midi_path: str):
        """Run MT3 transcription on one audio file and save to MIDI."""
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        est_ns = self(audio)
        note_seq.sequence_proto_to_midi_file(est_ns, output_midi_path)


# -------------------------------------------------------------------------
# STAGE 2: MIDI → Symbolic Features
# -------------------------------------------------------------------------
class MT3FeatureExtractor(ExtractorBase):
    """Extract symbolic note events from MT3-generated MIDI files."""

    name = "mt3"

    def __init__(self, checkpoint_dir: str, gin_dir: str, model_type: str = "mt3"):
        """Initialize MT3 transcriber and feature extractor."""
        self.transcriber = MT3Transcriber(checkpoint_dir, gin_dir, model_type)

    def _extract_note_events(self, midi_path: str) -> List[Dict[str, Any]]:
        """Extract symbolic note events from a MIDI file."""
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            events = []
            for inst in midi.instruments:
                if inst.is_drum:
                    continue
                name = program_to_instrument_name(inst.program)
                for note in inst.notes:
                    note_name = note_number_to_name(note.pitch)
                    events.append(
                        {
                            "instrument": name,
                            "pitch": note.pitch,
                            "note": note_name,
                            "start": round(note.start, 2),
                            "end": round(note.end, 2),
                        }
                    )
            return events
        except Exception as e:
            return [{"error": str(e)}]

    def run(self, audio_path: str, midi_out_dir: str) -> Dict[str, Any]:
        """Transcribe an audio file and return extracted symbolic note features."""
        os.makedirs(midi_out_dir, exist_ok=True)
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        midi_path = os.path.join(midi_out_dir, f"{file_id}.mid")

        try:
            self.transcriber.transcribe_file(audio_path, midi_path)
            events = self._extract_note_events(midi_path)
            return {"file_id": file_id, "events": events}
        except Exception as e:
            return {"file_id": file_id, "error": str(e)}

    def process_dir(self, input_dir: str, output_json: str, midi_out_dir: str) -> str:
        """Transcribe all audio files in a directory and export features to JSON."""
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        os.makedirs(midi_out_dir, exist_ok=True)

        results = {}
        for fname in tqdm(sorted(os.listdir(input_dir)), desc="MT3 transcription"):
            if not fname.lower().endswith((".wav", ".mp3", ".flac")):
                continue
            audio_path = os.path.join(input_dir, fname)
            entry = self.run(audio_path, midi_out_dir)
            results[entry["file_id"]] = entry.get("events", entry.get("error", []))

            # Save incrementally
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return output_json
