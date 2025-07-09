# Machine Learning Institute - Week 5 - Audio is all you need

This week, we are experimenting with sound and text and fine-tuning the Whisper model.

We will try using a CNN and transformer based model for sound classification, then switch to fine-tuning Whisper on a choice of tasks.

# Set-up

* Install the [git lfs](https://git-lfs.com/) extension **before cloning this repository**
* Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

Then install dependencies with:

```bash
uv sync --all-packages --dev
```

# Model Training

Run the following, with an optional `--model "model_name"` parameter

```bash
uv run -m model.start_train
```

# Run streamlit app

```bash
uv run streamlit run streamlit/app.py
```

# Ideas for presentation

* Generating music from a score and style "sing this as a choir"
* Try to identify / describe the speaker of different segments of audio
  * Maybe we can find a dataset or splice together dataset annotations
* Add emotional labels to a dataset
  * We'd need to find a good dataset
* Can we have it hear a song sung and output the notes of the track (e.g. in MIDI format)
  * Maybe we can find an autotuned dataset