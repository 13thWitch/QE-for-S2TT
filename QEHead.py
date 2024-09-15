from sacrebleu.metrics import BLEU, CHRF, TER
from comet import load_from_checkpoint, download_model

class QEHead:
    def __init__(self, weights):
        self.weights = weights
        self.metrics = {
            "bleu": lambda x, y: bleu(x, y), 
            "chrf": lambda x, y: chrf(x, y),
            "ter": lambda x, y: ter(x, y),
        }
        # To use reference-free comet as metric, add huggingface liscence key: https://github.com/Unbabel/COMET/blob/master/LICENSE.models.md
        # comet_path = download_model("Unbabel/wmt22-cometkiwi-da")
        # self.comet_model = load_from_checkpoint(comet_path)

    def get_QE_score(self, predictions, metric="bleu", interpret_as_corpus=False):
        """
        Calculate the QE score of the given predictions using the given metric.
        @param predictions: Dict of perturbation strategies and corresponding predictions to compare
        @param metric: Metric to use for QE calculation, default is COMET
        @return: QE score of the predictions, values 0 - 100
        """
        original = predictions["original"]
        perturbed_predictions = {key: value for key, value in predictions.items() if key != "original"}

        # If the predictions are to be interpreted as a corpus, use all predictions as reference text
        if interpret_as_corpus:
            return self.metrics[metric.lower()](original, list(perturbed_predictions.values()))

        deviations = self.get_deviations(perturbed_predictions, original, metric)
        raw_score = self.weighted_average(deviations)

        return self.normalize(raw_score, deviations.keys())

    def get_deviations(self, predictions, original, metric="bleu"):
        """
        Calculate each predictions deviation from the original prediction using the given metric.
        @param predictions: Dict of tagged predictions to compare
        @param original: Original prediction to compare to
        @param metric: Metric to use for deviation calculation, default is COMET
        @return: dict of deviations for each prediction
        """
        metric = self.metrics[metric.lower()]
        deviations = dict()
        for tag, prediction in predictions.items():
            deviations[tag] = metric(original, [prediction])
        return deviations
    
    def weighted_average(self, scores):
        """
        Aggregate the given scores using the QE weights.
        @param scores: dict of tags and their qualscores to aggregate
        @return: weighted aggregated deviations
        """
        weighted_scores = [value * self.weights[key] if key in self.weights else value for key, value in scores.items()]
        return sum(weighted_scores) / len(weighted_scores)
    
    def normalize(self, score, prediction_tags):
        """
        Normalize the given score to be between 0 and 100.
        @param score: float raw score
        @param prediction_tags: List of tags used in the prediction
        @return: float normalized score
        """
        used_weights = [self.weights[tag] for tag in prediction_tags if tag in self.weights]
        num_unweighted = len(prediction_tags) - len(used_weights)
        max_score = (sum(used_weights) + num_unweighted) / len(prediction_tags)
        return (score / max_score)   

def bleu(hypothesis, references):
    """
    Calculate the BLEU score between two sentences.
    @param hypothesis: First sentence to compare
    @param reference: Second sentence to compare, reference text
    @return: BLEU score between the two sentences
    """
    bleu = BLEU()
    bleu.effective_order = True
    return bleu.sentence_score(hypothesis=hypothesis, references=references).score

def chrf(hypothesis, references):
    """
    Calculate the CHRF score between two sentences.
    @param hypothesis: First sentence to compare
    @param references: List of reference sentences
    @return: CHRF score between the two sentences
    """
    chrf = CHRF()
    return chrf.sentence_score(hypothesis=hypothesis, references=references).score

def ter(hypothesis, references):
    """
    Calculate the inverse TER between two sentences. 
    Caution! TER returns [0, 100]. If TER score is more than 100, it is treated as 100.
    @param hypothesis: First sentence to compare
    @param references: List of reference sentences
    @return: TER score between the two sentences
    """
    ter = TER()
    ter_score = ter.sentence_score(hypothesis=hypothesis, references=references).score
    if ter_score > 100.0:
        ter_score = 100.0
    return (100 - ter_score)