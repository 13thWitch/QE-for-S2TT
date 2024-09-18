# QE-for-S2TT

This repository contains the implementation of an unsupervised, black-box method for Quality Estimation of Speech-to-Text Translation systems. It is perturbation based and uses prediction robustness under manipulation of the source audio as an indicator for quality.

## Installation

1. Set up a virtual environment including Python >= 3.11.
2. Use a package manager (e.g. pip) to install the requirements for your system. For a 64-bit Windows machine without NVIDIA CUDA run `pip install -r requirements_windows_noCuda.txt`. For the bwUniCluster (v2), a linux machine with CUDA, do the same using `requirements_cluster.txt`.

## Inference

To run inference, you need

- a valid model key and
- a path to an audio file.

You _may_ additionally configure

- source and target languages (the language keys you give must be compatible with the model you pass)
- desired metric to use (out of BLEU, TER and CHRF)
- as_corpus [boolean]: whether you'd like pairwise or corpus-like translation similarity calculation
- a path to a config file.

```
python ./inference.py \
 --model facebook/seamless-m4t-v2-large \
 --audio ./audio_original.wav \
 --source_lang por \
 --target_lang deu
```

### Models

The pre-embedded models are OpenAI's Whisper (`key: whisper`), Meta's SeamlessM4T-v2 (`key: facebook/seamless-m4t-v2-large`)and a trivial predictor (`key: stupid_model`). Manually adding models requires a model implementation and inserting model loading and inference methods in the designated sections of `ModelWrapper.py`.
Be advised that each model may vary in the mainfestation of its language keys (e.g. `whisper: 'en'`, but `seamless: eng`).

### Configuration

A configuration cosists of

- the metric (choice of BLEU, TER and CHRF)
- as_corpus (whether corpuslike translation similarity calculation should be applied)
- a dict/JSON Object of weights (contents will be ignored if corpuslike, if pairwise then an empty dict means all weights are equal)
- a dict/JSON Object of perturbations to perform.

This repository contains multiple example perturbations for reference.

#### Perturbation Specification

The perturbation specification is a dict/JSON object. It may contain the keys `resampling`, `random_noise`, `frequency_filtering` and `speed_warp`. The values of the respective keys have to correspond to the respective perturbation method's specifications. An example os provided below.

```
{
    "perturbation": {
        "resampling": {
            "target_sample_rates": [21000, 25000]
        },
        "frequency_filtering": {
            "stop_cutoffs": [
                [100, 3000],
                [50, 200]
            ],
            "pass_cutoffs": [
                [300, 5000],
                [1000, 7000]
            ],
        },
        "random_noise": {
            "std_ns": [0.01, 0.1]
        },
        "speed_warp": {
            "speeds": [0.7, 1.7]
        }
    }
}
```

The weight keys are given as `<perturbation_key>-<spec>`. For the example above this results in `random_noise-0.1` and `frequency_filtering-pass(300,5000)` (among others).
