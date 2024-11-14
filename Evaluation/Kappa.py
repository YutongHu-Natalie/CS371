import numpy as np
from typing import List, Union
import pandas as pd


class CohenKappaCalculator:
    """A class to calculate Cohen's Kappa statistic for measuring inter-rater reliability."""

    def __init__(self):
        self.kappa_score = None
        self.confusion_matrix = None
        self.observed_agreement = None
        self.expected_agreement = None

    def calculate_kappa(self, rater1: Union[List, np.ndarray], rater2: Union[List, np.ndarray]) -> float:
        """
        Calculate Cohen's Kappa score for two raters.

        Args:
            rater1: Ratings from first rater
            rater2: Ratings from second rater

        Returns:
            float: Cohen's Kappa score

        Raises:
            ValueError: If inputs are not of equal length or contain invalid values
        """
        # Convert inputs to numpy arrays
        rater1 = np.array(rater1)
        rater2 = np.array(rater2)

        # Validate inputs
        if len(rater1) != len(rater2):
            raise ValueError("Both raters must have the same number of ratings")

        if len(rater1) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Create confusion matrix
        unique_ratings = np.unique(np.concatenate([rater1, rater2]))
        n_categories = len(unique_ratings)
        self.confusion_matrix = np.zeros((n_categories, n_categories))

        # Fill confusion matrix
        for i in range(len(rater1)):
            r1_idx = np.where(unique_ratings == rater1[i])[0][0]
            r2_idx = np.where(unique_ratings == rater2[i])[0][0]
            self.confusion_matrix[r1_idx, r2_idx] += 1

        # Calculate observed agreement
        self.observed_agreement = np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

        # Calculate expected agreement
        row_sums = np.sum(self.confusion_matrix, axis=1)
        col_sums = np.sum(self.confusion_matrix, axis=0)
        total = np.sum(self.confusion_matrix)

        self.expected_agreement = np.sum((row_sums * col_sums) / total) / total

        # Calculate kappa
        self.kappa_score = (self.observed_agreement - self.expected_agreement) / (1 - self.expected_agreement)

        return self.kappa_score

    def get_metrics(self) -> dict:
        """
        Get all calculated metrics.

        Returns:
            dict: Dictionary containing kappa score, observed agreement, and expected agreement
        """
        if self.kappa_score is None:
            raise ValueError("Must calculate kappa before getting metrics")

        return {
            'kappa_score': self.kappa_score,
            'observed_agreement': self.observed_agreement,
            'expected_agreement': self.expected_agreement,
            'confusion_matrix': self.confusion_matrix
        }

    def interpret_kappa(self) -> str:
        """
        Interpret the strength of agreement based on Kappa score.

        Returns:
            str: Interpretation of kappa score
        """
        if self.kappa_score is None:
            raise ValueError("Must calculate kappa before interpreting")

        if self.kappa_score < 0:
            return "Poor agreement (less than chance)"
        elif self.kappa_score < 0.20:
            return "Slight agreement"
        elif self.kappa_score < 0.40:
            return "Fair agreement"
        elif self.kappa_score < 0.60:
            return "Moderate agreement"
        elif self.kappa_score < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

