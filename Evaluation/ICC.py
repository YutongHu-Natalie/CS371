import numpy as np
from scipy import stats
class ICCCalculator:

    """Calculator for Intraclass Correlation Coefficient."""

    def __init__(self):
        self.icc_score = None
        self.ci_lower = None
        self.ci_upper = None
        self.p_value = None

    def calculate(self,
                  ratings: np.ndarray,
                  icc_type: str = '2,1',
                  confidence: float = 0.95) -> float:
        """
        Calculate ICC for multiple raters.

        Args:
            ratings: Matrix of ratings (items × raters)
            icc_type: Type of ICC to calculate ('1,1', '2,1', '3,1', '1,k', '2,k', '3,k')
            confidence: Confidence level for intervals

        Returns:
            float: ICC score
        """
        ratings = np.array(ratings)

        if ratings.ndim != 2:
            raise ValueError("Ratings must be a 2D array (items × raters)")

        n_items, n_raters = ratings.shape

        if n_items < 2:
            raise ValueError("Need at least 2 items")
        if n_raters < 2:
            raise ValueError("Need at least 2 raters")

        # Calculate mean squares
        mean_by_items = np.mean(ratings, axis=1)
        mean_by_raters = np.mean(ratings, axis=0)
        grand_mean = np.mean(ratings)

        # Within-items sum of squares
        within_item_ss = np.sum((ratings - mean_by_items.reshape(-1, 1)) ** 2)

        # Between-items sum of squares
        between_items_ss = n_raters * np.sum((mean_by_items - grand_mean) ** 2)

        # Between-raters sum of squares
        between_raters_ss = n_items * np.sum((mean_by_raters - grand_mean) ** 2)

        # Calculate degrees of freedom
        dof_within = (n_items - 1) * (n_raters - 1)
        dof_between_items = n_items - 1
        dof_between_raters = n_raters - 1

        # Calculate mean squares
        ms_within = within_item_ss / dof_within
        ms_between_items = between_items_ss / dof_between_items
        ms_between_raters = between_raters_ss / dof_between_raters

        # Calculate ICC based on type
        if icc_type == '2,1':
            # Two-way random effects, single measures
            self.icc_score = (ms_between_items - ms_within) / (
                        ms_between_items + (n_raters - 1) * ms_within + n_raters * (
                            ms_between_raters - ms_within) / n_items)
        elif icc_type == '2,k':
            # Two-way random effects, average measures
            self.icc_score = (ms_between_items - ms_within) / ms_between_items
        else:
            raise ValueError("Unsupported ICC type. Use '2,1' or '2,k'")

        # Calculate confidence intervals using F-distribution
        f = ms_between_items / ms_within
        df1 = n_items - 1
        df2 = (n_items - 1) * (n_raters - 1)

        alpha = 1 - confidence
        f_lower = stats.f.ppf(alpha / 2, df1, df2)
        f_upper = stats.f.ppf(1 - alpha / 2, df1, df2)

        self.ci_lower = (f_lower * self.icc_score - 1) / (f_lower - 1)
        self.ci_upper = (f_upper * self.icc_score - 1) / (f_upper - 1)

        # Calculate p-value
        f_obs = (1 + (n_raters * self.icc_score) / (1 - self.icc_score))
        self.p_value = 1 - stats.f.cdf(f_obs, df1, df2)

        return self.icc_score

    def get_metrics(self) :
        """Get all calculated metrics."""
        if self.icc_score is None:
            raise ValueError("Must calculate ICC before getting metrics")

        return {
            'icc_score': self.icc_score,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'p_value': self.p_value
        }
