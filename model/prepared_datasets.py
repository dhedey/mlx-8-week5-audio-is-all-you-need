import datasets
import os
import huggingface_hub
import torchaudio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional

def noop_collate(batch):
    return batch

def resample_and_trim_or_pad(soundfile, target_sample_rate, target_num_samples) -> torch.Tensor:
    soundfile_array, sample_rate = soundfile["array"], soundfile["sampling_rate"]

    if soundfile_array.ndim == 1:
        waveform = torch.from_numpy(soundfile_array).to(torch.float).unsqueeze(0)  # [1, time]
    else:
        waveform = torch.from_numpy(soundfile_array).to(torch.float).T  # [channels, time]

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Trim or pad
    num_samples = waveform.shape[-1]
    if num_samples > target_num_samples:
        waveform = waveform[..., :target_num_samples]
    elif num_samples < target_num_samples:
        waveform = F.pad(waveform, (0, target_num_samples - num_samples))

    return waveform

def urbansound8K_prepare_v2(dataset_item, num_mels: Optional[int] = None, total_time_frames: Optional[int] = None):
    if num_mels is None:
        num_mels = 64
    if total_time_frames is None:
        # SAMPLE_RATE = 16000 corresponds to total_time_periods = 321
        total_time_frames = 321

    FIXED_LENGTH_SECONDS = 4

    # We want that:
    # * total_samples = sample_rate * 4 seconds
    # * total_samples = 200 * (total_time_frames - 1) + 400
    # sample_rate = ((total_time_frames - 1) * 200 + 400) / 4

    target_num_samples = 200 * (total_time_frames - 1) # + 400
    target_sample_rate = target_num_samples // FIXED_LENGTH_SECONDS

    waveform = resample_and_trim_or_pad(
        soundfile=dataset_item["audio"],
        target_sample_rate=target_sample_rate,
        target_num_samples=target_num_samples,
    )

    # We have:
    # * total_samples = 4s * 16000 samples / second
    # * n_fft = 400                          (the number of samples for the fft)
    # * window_length = n_fft = 400          (the number of samples for each window)
    # * hop_length = window_size / 2 = 200   (the stride / num of samples between each window)
    #
    # So we end up with the following number of time buckets:
    # 1 + (total_samples - window_length) / hop_length = 1 + (4 * 16000 - 400) / 200 = 319
    # ALTHOUGH it looks like it's actually:
    # 1 + total_samples / hop_length = 1 + (4 * 16000) / 200 = 321
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_mels=num_mels,
    )
    spectrogram = transform(waveform)

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
    data_folder = os.path.join(os.path.dirname(__file__), "datasets")

    assert 1 <= validation_fold <= 10

    dataset = datasets.load_dataset(
        "danavery/urbansound8K",
        cache_dir=data_folder,
        # It only comes with one split "train" but advises we do cross validation based on the fold
        split="train",
    )

    train_dataset = dataset.filter(lambda x: x["fold"] != validation_fold)
    train_dataset = train_dataset.map(lambda x: urbansound8K_prepare_v2(x, num_mels=num_mels, total_time_frames=total_time_frames), remove_columns=dataset.column_names)
    eval_dataset = dataset.filter(lambda x: x["fold"] == validation_fold)
    eval_dataset = eval_dataset.map(lambda x: urbansound8K_prepare_v2(x, num_mels=num_mels, total_time_frames=total_time_frames), remove_columns=dataset.column_names)

    return train_dataset, eval_dataset

if __name__ == "__main__":
    generate_urban_classifier_dataset(1)