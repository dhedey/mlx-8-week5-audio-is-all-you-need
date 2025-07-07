import datasets
import os
import huggingface_hub
import torchaudio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def noop_collate(batch):
    return batch

def resample_and_trim_or_pad(soundfile, target_sample_rate, target_length_seconds) -> torch.Tensor:
    soundfile_array, sample_rate = soundfile["array"], soundfile["sampling_rate"]

    if soundfile_array.ndim == 1:
        waveform = torch.from_numpy(soundfile_array).to(torch.float).unsqueeze(0)  # [1, time]
    else:
        waveform = torch.from_numpy(soundfile_array).to(torch.float).T  # [channels, time]

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Trim or pad
    target_num_samples = target_sample_rate * target_length_seconds
    num_samples = waveform.shape[-1]
    if num_samples > target_num_samples:
        waveform = waveform[..., :target_num_samples]
    elif num_samples < target_num_samples:
        waveform = F.pad(waveform, (0, target_num_samples - num_samples))

    return waveform

def urbansound8K_prepare(dataset_item):
    SAMPLE_RATE = 16000
    FIXED_LENGTH_SECONDS = 4

    waveform = resample_and_trim_or_pad(
        soundfile=dataset_item["audio"],
        target_sample_rate=SAMPLE_RATE,
        target_length_seconds=FIXED_LENGTH_SECONDS,
    )

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE)
    spectrogram = transform(waveform)

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

def generate_urban_classifier_dataset(validation_fold: int):
    data_folder = os.path.join(os.path.dirname(__file__), "datasets")

    assert 1 <= validation_fold <= 10

    dataset = datasets.load_dataset(
        "danavery/urbansound8K",
        cache_dir=data_folder,
        # It only comes with one split "train" but advises we do cross validation based on the fold
        split="train",
    )

    train_dataset = dataset.filter(lambda x: x["fold"] != validation_fold)
    train_dataset = train_dataset.map(urbansound8K_prepare, remove_columns=dataset.column_names)
    eval_dataset = dataset.filter(lambda x: x["fold"] == validation_fold)
    eval_dataset = eval_dataset.map(urbansound8K_prepare, remove_columns=dataset.column_names)

    return train_dataset, eval_dataset

if __name__ == "__main__":
    generate_urban_classifier_dataset(1)