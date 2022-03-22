import torch
from speechbrain.utils.metric_stats import MetricStats

class SimilarityMetricsStats(MetricStats):
    """Aggregates the similarity for the encoder output of the ASR to track utility degradation
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.scores = []
        self.summary = {}

    def append(self, scores):
        """Appends scores and labels to internal lists.

        Does not compute metrics until time of summary, since
        automatic thresholds (e.g., EER) need full set of scores.

        Arguments
        ---------
        ids : list
            The string ids for the samples

        """
        self.scores.extend(scores.detach())

    def summarize(
        self
    ):
        """
        Arguments
        ---------
        field : str
            A key for selecting a single statistic. If not provided,
            a dict with all statistics is returned.
        threshold : float
            If no threshold is provided, equal error rate is used.
        """

        if isinstance(self.scores, list):
            self.scores = torch.stack(self.scores)

        self.summary["average"] = torch.sum(self.scores)/self.scores.shape[0]
        return self.summary["average"]