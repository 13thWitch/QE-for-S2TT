perturbation = {
    "random_noise": {
                    "std_ns": [
                        0.003,
                        0.005,
                        0.007,
                        0.009,
                        0.012,
                        0.015
                    ]
                },
}

weights = {
                "random_noise-0.001": 0.7,
                "random_noise-0.002": 0.8,
                "random_noise-0.003": 0.9,
                "random_noise-0.004": 1.0,
                "random_noise-0.005": 1.3,
                "random_noise-0.006": 1.6,
                "random_noise-0.007": 1.9,
                "random_noise-0.009": 2.0,
                "random_noise-0.012": 2.1,
                "random_noise-0.015": 2.2,
        }

metric = "bleu"
as_corpus = False


"""config = {
        "perturbation": {
            "random_noise": {
                "std_ns": [0.001, 0.005, 0.007]
            }, 
            "resampling": {
                "target_sample_rates": [8000, 32000]
            },
            "speed_warp": {
                "speeds": [0.7, 1.7]
            }, 
            "frequency_filtering": {
                "pass_cutoffs": [(100, 3000)],
            }
        }, 
    "weights": {
            "random_noise-0.001": 2,
            "random_noise-0.005": 1.7,
            "random_noise-0.007": 1.5,
            "resampling-8000": 0.3,
            "resampling-32000": 0.3,
        },
    }
"""