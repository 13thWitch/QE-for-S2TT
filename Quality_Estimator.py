from ModelWrapper import STModel
from Perturbation import Perturbator
from QEHead import QEHead
import torch
import json

class QualityEstimator:
    def __init__(self, model, source_lang, target_lang):
        self.model = STModel(model, target_language=target_lang, source_language=source_lang)
        self.config = {
            "random_noise": {
                "std_ns": [0.001]
            }, 
            "resampling": {
                "target_sample_rates": [8000, 32000]
            },
            "speed_warp": {
                "speeds": [0.7, 1, 1.7]
            }, 
            "frequency_filtering": {
                "pass_cutoffs": [(300, 3000), (1000, 10000)],
                "stop_cutoffs": [(300, 3000), (1000, 10000)]
            }
        }
        self.perturbator = Perturbator(self.config)
        self.weights = {}
        self.QEHead = QEHead(weights=self.weights)

    def estimate_quality(self, audio, sample_rate, eval=False):
        """
        Estimate the quality of the model's translation of the audio. Returns additional info if eval=True.
        @param audio: audio file to be translated
        @param sample_rate: sample rate of the audio file
        @param eval: whether to return the confidence of the translation
        @return: BLEU score of the translation, and additional info if eval=True
        """
        perturbations = self.perturbator.get_perturbations(audio=audio, sample_rate=sample_rate)
        perturbations["original"] = audio
        if not eval:
            try: 
                predictions = self.model.infer(perturbations, sample_rate, return_confidence=eval)
                score = self.QEHead.get_QE_score(predictions=predictions, metric="bleu", interpret_as_corpus=False)
                return score
            except:
                return float('NaN')

        try:
            predictions, likelihoods = self.model.infer(perturbations, sample_rate, return_confidence=eval)
            score = self.QEHead.get_QE_score(predictions=predictions, metric="bleu", interpret_as_corpus=False)
            eval_data = {
                "translation": predictions['original'],
                "confidence": likelihoods['original'],
            }
            return score, eval_data
        except:
            return float('NaN'), {"translation": None, "confidence": float("NaN")}
        
    
    def document(self, result, reference_QE=None):
        """
        Save the run information, including result (and expected result if available) to a json file.
        """
        cummulated_run_info = {
            "config": self.config,
            "weights": self.weights
        }
        cummulated_run_info["result"] = result
        if reference_QE is not None:
            cummulated_run_info["reference"] = reference_QE
        
        with open("documentation.json", "w", encoding="utf-8") as f:
            json.dump(cummulated_run_info, f, indent=4, ensure_ascii=False)