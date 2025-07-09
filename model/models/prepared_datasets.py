import datasets
import os
import huggingface_hub
import torchaudio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional

from torchgen.utils import OrderedSet
from transformers import WhisperProcessor, WhisperModel
import torch
import torchaudio
from model.harness import datasets_cache_folder, select_device, select_device_no_mps

def noop_collate(batch):
    return batch

def resample_and_trim_or_pad(soundfile, target_sample_rate, target_num_samples) -> torch.Tensor:
    soundfile_array, sample_rate = soundfile["array"], soundfile["sampling_rate"]

    if soundfile_array.ndim == 1:
        waveform = torch.from_numpy(soundfile_array).to(torch.float32).unsqueeze(0)  # [1, time]
    else:
        waveform = torch.from_numpy(soundfile_array).to(torch.float32).T  # [channels, time]

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate).to('cpu')
        waveform = resampler(waveform)

    # Trim or pad
    num_samples = waveform.shape[-1]
    if num_samples > target_num_samples:
        waveform = waveform[..., :target_num_samples]
    elif num_samples < target_num_samples:
        waveform = F.pad(waveform, (0, target_num_samples - num_samples))

    return waveform.to('cpu')

def urbansound8K_prepare_v2(dataset_item, num_mels: Optional[int] = None, total_time_frames: Optional[int] = None):
    if num_mels is None:
        num_mels = 64
    if total_time_frames is None:
        # SAMPLE_RATE = 16000 corresponds to total_time_periods = 321
        total_time_frames = 321

    FIXED_LENGTH_SECONDS = 4

    target_num_samples = 200 * (total_time_frames - 1) # + 400
    target_sample_rate = target_num_samples // FIXED_LENGTH_SECONDS

    waveform = resample_and_trim_or_pad(
        soundfile=dataset_item["audio"],
        target_sample_rate=target_sample_rate,
        target_num_samples=target_num_samples,
    )

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_mels=num_mels,
    ).to('cpu')  # Ensure transform is on CPU
    spectrogram = transform(waveform)
    spectrogram = spectrogram.to('cpu')

    expected_shape = (1, num_mels, total_time_frames) # 1 = channel
    assert spectrogram.shape == expected_shape, f"Expected spectrogram to have shape {expected_shape}, but got {spectrogram.shape}"

    # elapsed_seconds = dataset_item["end"] - dataset_item["start"]
    #
    # plt.figure(figsize=(10, 4))
    # plt.imshow(spectrogram.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='magma')
    # plt.colorbar(format='%+2.0f dB')
    # plt.xlabel('Time')
    # plt.ylabel('Mel Frequency Bin')
    # plt.title(f'Mel Spectrogram ({elapsed_seconds}/4 seconds)')
    # plt.tight_layout()
    # plt.show()

    return {
        "spectrogram": spectrogram,
        "class_id": dataset_item["classID"],
        "class": dataset_item["class"],
        "salience": dataset_item["salience"],
    }

def generate_urban_classifier_dataset(validation_fold: int, num_mels: Optional[int] = None, total_time_frames: Optional[int] = None):
    assert 1 <= validation_fold <= 10

    dataset = datasets.load_dataset(
        "danavery/urbansound8K",
        cache_dir=datasets_cache_folder(),
        # It only comes with one split "train" but advises we do cross validation based on the fold
        split="train",
    )

    train_dataset = dataset.filter(lambda x: x["fold"] != validation_fold)
    train_dataset = train_dataset.map(lambda x: urbansound8K_prepare_v2(x, num_mels=num_mels, total_time_frames=total_time_frames), remove_columns=dataset.column_names)
    eval_dataset = dataset.filter(lambda x: x["fold"] == validation_fold)
    eval_dataset = eval_dataset.map(lambda x: urbansound8K_prepare_v2(x, num_mels=num_mels, total_time_frames=total_time_frames), remove_columns=dataset.column_names)

    return train_dataset, eval_dataset

_preloaded_whisper = None

def get_whisper() -> tuple[WhisperProcessor, WhisperModel]:
    global _preloaded_whisper
    if _preloaded_whisper is None:
        print("Loading whisper...")
        device = select_device_no_mps()
        _preloaded_whisper = (WhisperProcessor.from_pretrained("openai/whisper-tiny"), WhisperModel.from_pretrained("openai/whisper-tiny").to(device))
    return _preloaded_whisper

_vctk_speaker_id_mapping = {}

def prepare_vctk(item):
    global _vctk_speaker_id_mapping
    soundfile = item["flac"]

    # MPS is not supported by Whisper - we get:
    # > NotImplementedError: Output channels > 65536 not supported at the MPS device.
    device = select_device_no_mps()

    waveform_np, sample_rate = soundfile["array"], soundfile["sampling_rate"]
    waveform = torch.from_numpy(waveform_np).to(torch.float).to(device)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    # Whisper expects mono audio
    if waveform.shape[0] > 1:
       waveform = waveform.mean(dim=0, keepdim=True)

    # TODO: Add in randomization of the audio

    # Load model and processor
    processor, model = get_whisper()

    # Prepare input for Whisper
    inputs = processor(waveform.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt")

    # Get encoder hidden states (embeddings)
    with torch.no_grad():
        encoder_outputs = model.encoder(inputs.input_features.to(device))

    # encoder_outputs.last_hidden_state shape: [batch, time, hidden_dim]
    # Note: Outputs 50 embeddings / sec, over 30 seconds for a total of 1500
    whisper_embedding = encoder_outputs.last_hidden_state  # This is the sequence of embeddings

    speaker_id = item["speaker_id"]
    if speaker_id not in _vctk_speaker_id_mapping:
        _vctk_speaker_id_mapping[speaker_id] = len(_vctk_speaker_id_mapping)

    speaker_index = _vctk_speaker_id_mapping[speaker_id]

    return {
        "whisper_embedding": whisper_embedding, # [batch, time, hidden_dim]
        "speaker_index": speaker_index,
    }

_counter = 0
def every_100th(item):
    global _counter
    is_included = (_counter % 100 == 0)
    _counter += 1
    return is_included

def every_10th(item):
    global _counter
    is_included = (_counter % 10 == 0)
    _counter += 1
    return is_included

def generate_speaker_tagged_dataset():
    dataset = datasets.load_dataset("badayvedat/VCTK", cache_dir=datasets_cache_folder())

    get_whisper() # Pre-load Whisper

    eval = dataset["validation"].filter(every_10th).map(prepare_vctk, remove_columns=dataset["validation"].column_names)
    _counter = 0
    train = dataset["train"].filter(every_10th).map(prepare_vctk, remove_columns=dataset["train"].column_names)

    return train, eval, len(_vctk_speaker_id_mapping)


if __name__ == "__main__":
    generate_speaker_tagged_dataset()
    # generate_urban_classifier_dataset(1)