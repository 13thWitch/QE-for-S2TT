from ModelWrapper import STModel
from Perturbation import Perturbator
from QEHead import QEHead
import default_config
import json

class QualityEstimator:
    def __init__(self, model, source_lang, target_lang, perturbation={}, weights={}, metric="bleu", as_corpus=False):
        """
        Constructor of the Quality Estimator class.
        @param model: string key of model, choice of "whisper" and "huggingface"
        @param source_lang: string language key of source language
        @param target_lang: string language key of target language
        @param [optional] perturbation: dict of perturbations to perform with corresponding specifications. If empty or unset, default is used.
        @param [optional] weights: dict of weights to assign the perturbed translations. If empty or unset, default is used.
        @param [optional] metric: comparative metric to use for translation comparison
        @param [optional] as_corpus: should the translation similarity be calculated corpus-like?
        """
        self.model = STModel(model, target_language=target_lang, source_language=source_lang)
        # Options: 
        # frequency_filtering with spec tuple array pass_cutoffs
        # resampling with spec array target_sample_rates
        # random_noise with spec array std_ns
        # speed_warp with spec array speeds
        self.config = perturbation if perturbation else default_config.perturbation
        self.perturbator = Perturbator(self.config)
        # Keylog for weights
        # frequency_filtering-{stop, pass}(lower, upper)
        # resampling-newsr
        # random_noise-stdns
        # speed_warp-speed
        self.weights = weights if weights else default_config.weights
        self.QEHead = QEHead(weights=self.weights)
        self.metric = metric
        self.as_corpus = as_corpus

    def estimate_quality(self, audio, sample_rate, metric="", eval=False, as_corpus=False):
        """
        Estimate the quality of the model's translation of the audio. Returns additional info if eval=True.
        @param audio: audio file to be translated
        @param sample_rate: sample rate of the audio file
        @param metric: metric to use for quality estimation, uses default if not set
        @param eval: whether to return the confidence of the translation
        @param as_corpus: whether to interpret the translations as a corpus, default False
        @return: score of the translation, and additional info if eval=True. In case of failure returnes None values in correct output format
        """
        working_metric = metric if metric else self.metric
        working_as_corpus = as_corpus if as_corpus else self.as_corpus
        perturbations = self.perturbator.get_perturbations(audio=audio, sample_rate=sample_rate)
        perturbations["original"] = audio
        if not eval:
            try: 
                predictions = self.model.infer(perturbations, sample_rate, return_confidence=eval)
                score = self.QEHead.get_QE_score(predictions=predictions, metric=working_metric, interpret_as_corpus=working_as_corpus)
                return score
            except:
                return None

        try:
            predictions, likelihoods = self.model.infer(perturbations, sample_rate, return_confidence=eval)
            score = self.QEHead.get_QE_score(predictions=predictions, metric=working_metric, interpret_as_corpus=working_as_corpus)
            eval_data = {
                "translation": predictions['original'],
                "confidence": likelihoods['original'],
            }
            return score, eval_data
        except:
            return None, {"translation": None, "confidence": None}
        
    
    def document(self, result, reference_QE=None):
        """
        Save the run information, including result (and expected result if available) to a json file.
        """
        cummulated_run_info = {
            "config": self.config,
            "weights": self.weights,
            "metric": self.metric,
            "as_corpus": self.as_corpus
        }
        cummulated_run_info["result"] = result
        if reference_QE is not None:
            cummulated_run_info["reference"] = reference_QE
        
        with open("documentation.json", "w", encoding="utf-8") as f:
            json.dump(cummulated_run_info, f, indent=4, ensure_ascii=False)