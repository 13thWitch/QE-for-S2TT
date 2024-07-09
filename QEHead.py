class QEHead:
    def __init__(self, weights):
        self.weights = weights
        self.metrics = {
            "comet": lambda x: comet(x),
            "bleu": lambda x: bleu(x)
        }

    def get_QE_score(self, predictions, metric="COMET"):
        """
        Calculate the QE score of the given predictions using the given metric.
        @param predictions: List of predictions to compare
        @param metric: Metric to use for QE calculation, default is COMET
        @return: QE score of the predictions
        """
        original = predictions["original"]
        del predictions["original"]
        deviations = self.get_deviations(predictions, original, metric)
        raw_score = self.weighted_average(deviations)
        num_unweighted = len(deviations) - len(self.weights)
        return self.normalize(raw_score, offset=num_unweighted)

    def get_deviations(self, predictions, original, metric="COMET"):
        """
        Calculate each predictions deviation from the original prediction using the given metric.
        @param predictions: List of predictions to compare
        @param original: Original prediction to compare to
        @param metric: Metric to use for deviation calculation, default is COMET
        @return: dict of deviations for each prediction
        """
        metric = self.metrics[metric.lower()]
        deviations = dict()
        for tag, prediction in predictions.items():
            deviations[tag] = metric(prediction, original)
        return deviations
    
    def weighted_average(self, scores):
        """
        Aggregate the given scores using the QE weights.
        @param scores: dict of tags and their qualscores to aggregate
        @return: weighted aggregated deviations
        """
        result = 0
        for tag, score in scores.items():
            weight = self.weights[tag] if tag in self.weights else 1
            result += score * weight 
        return result / len(scores)
    
    def normalize(self, score, offset=0):
        """
        Normalize the given score to be between 0 and 1.
        @param score: float raw score
        @param offset: int count of unweighted scores
        @return: float normalized score
        """
        max_score = (sum(self.weights.values()) + offset) / (len(self.weights.values()) + offset)
        return score / max_score
    
def comet(sentence_1, sentence_2):
    """
    Calculate the COMET score between two sentences.
    @param sentence_1: First sentence to compare
    @param sentence_2: Second sentence to compare
    @return: COMET score between the two sentences
    """
    return 0.5

def bleu(sentence_1, sentence_2):
    """
    Calculate the BLEU score between two sentences.
    @param sentence_1: First sentence to compare
    @param sentence_2: Second sentence to compare
    @return: BLEU score between the two sentences
    """
    return 0.5