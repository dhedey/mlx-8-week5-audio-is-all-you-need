from model.models.speaker_embedding_model import SpeakerEmbeddingTwoTowers
from .project_config import DEFINED_MODELS

if __name__ == "__main__":
    from model.harness import ModelBase, TrainingOverrides, ModelTrainerBase, TrainingState

    for model_name in DEFINED_MODELS.keys():
        best_version = f"{model_name}-best"
        lookup_locations = [
            (best_version, "trained"),
            (model_name, "trained"),
            (best_version, "snapshots"),
            (model_name, "snapshots"),
        ]

        for name, location in lookup_locations:
            if ModelBase.exists(model_name=name, location=location):
                print(f"Loading Model: {location}/{name}")
                model, training_state_dict, training_config_dict = ModelBase.load_advanced(model_name=best_version)
                break
        else:
            print(f"Model {model_name} or {best_version} could not be found in the trained or snapshots folder. Skipping.")
            continue

        model = model.eval()
        model.print_detailed_parameter_counts()

        if isinstance(model, SpeakerEmbeddingTwoTowers):
            import soundfile as sf
            import os
            import numpy as np
            import sklearn
            import matplotlib.pyplot as plt
            import torchaudio

            embedding_model = model.speaker_embedding_model
            files = [
                ("andy-houston.flac", "andy", "houston"),
                ("andy-terminator-2.flac", "andy", "terminator-2"),
                ("andy-im-sorry.flac", "andy", "im-sorry"),
                ("api-houston.flac", "api", "houston"),
                ("api-terminator-2.flac", "api", "terminator-2"),
                ("api-im-sorry.flac", "api", "im-sorry"),
                ("david-houston.flac", "david", "houston"),
                ("david-terminator-2.flac", "david", "terminator-2"),
                ("david-im-sorry.flac", "david", "im-sorry"),
                ("nick-houston.flac", "nick", "houston"),
                ("nick-terminator-2.flac", "nick", "terminator-2"),
                ("nick-im-sorry.flac", "nick", "im-sorry"),
                ("noone-speaking.flac", "noone", "background"),
            ]
            embeddings = []
            labels = []
            speakers = []
            full_labels = []
            for file_name, speaker, context in files:
                print(f"Processing {file_name}")
                file_path = os.path.join(os.path.dirname(__file__), "samples", file_name)
                waveform_pt, sample_rate = torchaudio.load(file_path)
                whisper_embedding = embedding_model.process([
                    (waveform_pt.numpy(), sample_rate)
                ]).to(model.get_device())
                embedding = embedding_model(whisper_embedding).detach().mean(dim=1).squeeze(0).cpu().numpy()
                embeddings.append(embedding)
                speakers.append(speaker)
                labels.append(context)
                full_labels.append(f"{speaker}-{context}")

            embeddings = np.vstack(embeddings)

            print("Generating heatmap")
            sim_mat = sklearn.metrics.pairwise.cosine_similarity(embeddings, embeddings)  # (N, N)

            plt.figure(figsize=(8, 8))
            im = plt.imshow(sim_mat, interpolation='nearest', aspect='auto')
            plt.colorbar(im, fraction=0.046, pad=0.04)

            ticks = np.arange(len(labels))
            plt.xticks(ticks, full_labels, rotation=90)
            plt.yticks(ticks, full_labels)

            plt.title("Cosine Similarity of Speaker Embeddings")
            plt.tight_layout()
            plt.show()

            print("Generating TSNE...")
            tsne = sklearn.manifold.TSNE(n_components=2, perplexity=5, random_state=42)
            embedded_2d = tsne.fit_transform(embeddings)

            plt.figure(figsize=(8, 6))
            for speaker in set(speakers):
                idxs = [i for i, l in enumerate(speakers) if l == speaker]
                xs = embedded_2d[idxs, 0]
                ys = embedded_2d[idxs, 1]
                these_labels = [labels[id] for id in idxs]
                plt.scatter(xs, ys, label=speaker)
                for x, y, label in zip(xs, ys, these_labels):
                    plt.text(x + 2, y + 2, label, fontsize=8)

            plt.legend()
            plt.title("Speaker Embeddings (t-SNE)")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.show()

        trainer = ModelTrainerBase.load(
            model,
            training_config_dict,
            state=TrainingState.from_dict(training_state_dict),
            overrides=TrainingOverrides(
                override_batch_limit=1, # Speed up the validation pass below
            )
        )

        print(f"Latest validation metrics: {trainer.latest_validation_results}")

        print(f"Running model to check it's working...")
        trainer.validate()
