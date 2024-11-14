from typing import Union, List
import numpy as np

class KrippendorffAlphaCalculator:
    """Calculator for Krippendorff's Alpha."""

    def __init__(self):
        self.alpha_score = None
        self.confidence_interval = None
        self.n_bootstrap = 2000

    def _distance_metric(self, v1: Union[int, float], v2: Union[int, float], level: str = 'ordinal') -> float:
        """Calculate distance between two values based on measurement level."""
        if level == 'nominal':
            return 0 if v1 == v2 else 1
        elif level == 'ordinal':
            return (v1 - v2) ** 2
        else:
            raise ValueError("Unsupported level of measurement")

    def calculate(self,
                  ratings: Union[List[List], np.ndarray],
                  level: str = 'ordinal',
                  confidence: float = 0.95) -> float:
        """
        Calculate Krippendorff's Alpha.

        Args:
            ratings: Matrix of ratings (items × raters)
            level: Level of measurement ('nominal' or 'ordinal')
            confidence: Confidence level for bootstrap intervals

        Returns:
            float: Krippendorff's Alpha score
        """
        ratings = np.array(ratings)

        if ratings.ndim != 2:
            raise ValueError("Ratings must be a 2D array (items × raters)")

        n_items, n_raters = ratings.shape

        # Calculate observed disagreement
        observed_disagreement = 0
        n_pairs = 0

        for i in range(n_items):
            item_ratings = ratings[i][~np.isnan(ratings[i])]
            if len(item_ratings) >= 2:
                for j in range(len(item_ratings)):
                    for k in range(j + 1, len(item_ratings)):
                        observed_disagreement += self._distance_metric(item_ratings[j], item_ratings[k], level)
                        n_pairs += 1

        observed_disagreement /= n_pairs if n_pairs > 0 else 1

        # Calculate expected disagreement
        all_ratings = ratings.flatten()
        all_ratings = all_ratings[~np.isnan(all_ratings)]
        expected_disagreement = 0
        n_total_pairs = len(all_ratings) * (len(all_ratings) - 1) / 2

        for i in range(len(all_ratings)):
            for j in range(i + 1, len(all_ratings)):
                expected_disagreement += self._distance_metric(all_ratings[i], all_ratings[j], level)

        expected_disagreement /= n_total_pairs if n_total_pairs > 0 else 1

        # Calculate alpha
        self.alpha_score = 1 - (observed_disagreement / expected_disagreement if expected_disagreement != 0 else 0)

        # Bootstrap confidence intervals
        bootstrap_alphas = []

        for _ in range(self.n_bootstrap):
            # Resample items with replacement
            bootstrap_indices = np.random.choice(n_items, size=n_items, replace=True)
            bootstrap_ratings = ratings[bootstrap_indices]

            # Calculate alpha for bootstrap sample
            obs_d = 0
            n_p = 0

            for i in range(n_items):
                item_ratings = bootstrap_ratings[i][~np.isnan(bootstrap_ratings[i])]
                if len(item_ratings) >= 2:
                    for j in range(len(item_ratings)):
                        for k in range(j + 1, len(item_ratings)):
                            obs_d += self._distance_metric(item_ratings[j], item_ratings[k], level)
                            n_p += 1

            obs_d /= n_p if n_p > 0 else 1
            bootstrap_alphas.append(1 - (obs_d / expected_disagreement if expected_disagreement != 0 else 0))

        # Calculate confidence intervals
        alpha = (1 - confidence) / 2
        self.confidence_interval = (
            np.percentile(bootstrap_alphas, alpha * 100),
            np.percentile(bootstrap_alphas, (1 - alpha) * 100)
        )

        return self.alpha_score

    def get_metrics(self) :
        """Get all calculated metrics."""
        if self.alpha_score is None:
            raise ValueError("Must calculate alpha before getting metrics")

        return {
            'alpha_score': self.alpha_score,
            'confidence_interval': self.confidence_interval
        }