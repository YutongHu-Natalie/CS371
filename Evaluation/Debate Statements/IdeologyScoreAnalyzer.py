import pandas as pd
import numpy as np
from typing import Union, Dict
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


class IdeologyScoreAnalyzer:
    def __init__(self):
        self.data = None
        self.gold_standard = None

    def load_csv(self, filepath: str, start_id: str = None, end_id: str = None) -> None:
        try:
            print(f"Attempting to load file: {filepath}")
            self.data = pd.read_csv(filepath)
            print("Successfully loaded CSV file")

            # Ensure Statement ID is string with leading zeros
            self.data['Statement ID'] = self.data['Statement ID'].astype(str).str.zfill(3)

            # Set Statement ID as index
            self.data.set_index('Statement ID', inplace=True)

            # Filter by ID range if specified
            if start_id or end_id:
                start_id = str(start_id).zfill(3) if start_id else '000'
                end_id = str(end_id).zfill(3) if end_id else '999'
                self.data = self.data[
                    (self.data.index >= start_id) &
                    (self.data.index <= end_id)
                    ]
                print(f"Filtered data to ID range {start_id}-{end_id}")

            print("Dataset Info:")
            print(f"Columns in dataset: {list(self.data.columns)}")
            print(f"Number of rows: {len(self.data)}")

        except Exception as e:
            print(f"Error loading file: {e}")
            raise

    def _is_valid_score(self, score: Union[int, float]) -> bool:
        """Check if score is valid (0-10 range)"""
        try:
            score = float(score)
            return 0 <= score <= 10
        except:
            return False


    def analyze_human_agreement(self,
                                human1_col: str = 'human1',
                                human2_col: str = 'human2',
                                human3_col: str = 'human3',
                                overlap_ranges: Dict = None,
                                verbose: bool = True) -> Dict:
        """
        Analyze agreement between human annotators on 0-10 scale for specified overlap regions

        Args:
            human1_col, human2_col, human3_col: column names for human annotators
            overlap_ranges: dict with keys 'h1h2' and 'h2h3' containing tuples of (start_id, end_id)
            verbose: whether to print detailed results
        """
        results = {}

        def analyze_pair(scores1, scores2, pair_name):
            """Helper function to analyze a pair of annotators"""
            if len(scores1) == 0:
                print(f"No valid score pairs found for {pair_name}")
                return None

            # Calculate metrics
            pearson_r, pearson_p = stats.pearsonr(scores1, scores2)
            spearman_r, spearman_p = stats.spearmanr(scores1, scores2)
            icc = stats.pearsonr(scores1, scores2)[0]
            mse = mean_squared_error(scores1, scores2)
            rmse = np.sqrt(mse)
            agreement_1point = np.mean(np.abs(scores1 - scores2) <= 1)

            pair_results = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'icc': icc,
                'mse': mse,
                'rmse': rmse,
                'agreement_1point': agreement_1point,
                'n_samples': len(scores1),
                'scores1': scores1,
                'scores2': scores2
            }

            if verbose:
                print(f"\nResults for {pair_name}:")
                print("-" * 50)
                print(f"Number of valid samples: {pair_results['n_samples']}")
                print(f"Pearson correlation: {pair_results['pearson_r']:.3f} (p={pair_results['pearson_p']:.3f})")
                print(f"Spearman correlation: {pair_results['spearman_r']:.3f} (p={pair_results['spearman_p']:.3f})")
                print(f"ICC: {pair_results['icc']:.3f}")
                print(f"RMSE: {pair_results['rmse']:.3f}")
                print(f"Agreement within ±1 point: {pair_results['agreement_1point']:.3f}")

            return pair_results

        if overlap_ranges:
            # Analyze H1-H2 overlap
            h1h2_start, h1h2_end = overlap_ranges['h1h2']
            h1h2_mask = (self.data.index >= str(h1h2_start).zfill(3)) & (self.data.index <= str(h1h2_end).zfill(3))

            scores1_h1h2 = pd.to_numeric(self.data[human1_col][h1h2_mask], errors='coerce')
            scores2_h1h2 = pd.to_numeric(self.data[human2_col][h1h2_mask], errors='coerce')

            # Filter for valid scores (0-10)
            valid_mask_h1h2 = scores1_h1h2.between(0, 10) & scores2_h1h2.between(0, 10)
            results['h1h2'] = analyze_pair(
                scores1_h1h2[valid_mask_h1h2],
                scores2_h1h2[valid_mask_h1h2],
                "Human 1 vs Human 2"
            )

            # Analyze H2-H3 overlap
            h2h3_start, h2h3_end = overlap_ranges['h2h3']
            h2h3_mask = (self.data.index >= str(h2h3_start).zfill(3)) & (self.data.index <= str(h2h3_end).zfill(3))

            scores2_h2h3 = pd.to_numeric(self.data[human2_col][h2h3_mask], errors='coerce')
            scores3_h2h3 = pd.to_numeric(self.data[human3_col][h2h3_mask], errors='coerce')

            # Filter for valid scores (0-10)
            valid_mask_h2h3 = scores2_h2h3.between(0, 10) & scores3_h2h3.between(0, 10)
            results['h2h3'] = analyze_pair(
                scores2_h2h3[valid_mask_h2h3],
                scores3_h2h3[valid_mask_h2h3],
                "Human 2 vs Human 3"
            )

        return results

    def determine_gold_standard(self,
                                overlap_ranges: Dict,
                                strategy: str = 'mean',
                                verbose: bool = True) -> pd.Series:
        """
        Determine gold standard scores using specified strategy

        Args:
            overlap_ranges: dict with keys 'h1h2' and 'h2h3' containing tuples of (start_id, end_id)
            strategy: 'mean' or 'round' for handling overlapping annotations
            verbose: whether to print detailed results
        """
        if verbose:
            print(f"\nDetermining gold standard scores ({strategy} strategy)...")

        # Initialize gold standard series
        self.gold_standard = pd.Series(index=self.data.index, dtype=float)

        scores1 = pd.to_numeric(self.data['human1'], errors='coerce')
        scores1 = scores1.where(scores1 >= 0, np.nan)

        scores2 = pd.to_numeric(self.data['human2'], errors='coerce')
        scores2 = scores2.where(scores2 >= 0, np.nan)

        scores3 = pd.to_numeric(self.data['human3'], errors='coerce')
        scores3 = scores3.where(scores3 >= 0, np.nan)

        # Handle H1-H2 overlap region
        h1h2_start, h1h2_end = overlap_ranges['h1h2']
        h1h2_mask = (self.data.index >= str(h1h2_start).zfill(3)) & (self.data.index <= str(h1h2_end).zfill(3))

        # Handle H2-H3 overlap region
        h2h3_start, h2h3_end = overlap_ranges['h2h3']
        h2h3_mask = (self.data.index >= str(h2h3_start).zfill(3)) & (self.data.index <= str(h2h3_end).zfill(3))

        # Calculate means for overlap regions
        if strategy == 'mean':
            # H1-H2 overlap
            self.gold_standard[h1h2_mask] = np.mean([
                scores1[h1h2_mask],
                scores2[h1h2_mask]
            ], axis=0)

            # H2-H3 overlap
            self.gold_standard[h2h3_mask] = np.mean([
                scores2[h2h3_mask],
                scores3[h2h3_mask]
            ], axis=0)
        elif strategy == 'round':
            # H1-H2 overlap
            means_h1h2 = np.mean([scores1[h1h2_mask], scores2[h1h2_mask]], axis=0)
            self.gold_standard[h1h2_mask] = np.round(means_h1h2)

            # H2-H3 overlap
            means_h2h3 = np.mean([scores2[h2h3_mask], scores3[h2h3_mask]], axis=0)
            self.gold_standard[h2h3_mask] = np.round(means_h2h3)

        # For non-overlapping regions, use individual annotations
        for idx in self.data.index:
            if pd.isna(self.gold_standard[idx]):
                if not pd.isna(scores1[idx]) and self._is_valid_score(scores1[idx]):
                    self.gold_standard[idx] = scores1[idx]
                elif not pd.isna(scores2[idx]) and self._is_valid_score(scores2[idx]):
                    self.gold_standard[idx] = scores2[idx]
                elif not pd.isna(scores3[idx]) and self._is_valid_score(scores3[idx]):
                    self.gold_standard[idx] = scores3[idx]

        if verbose:
            print("\nGold standard statistics:")
            print(f"Total annotations: {len(self.gold_standard.dropna())}")
            print(f"From H1-H2 overlap ({h1h2_start}-{h1h2_end}): {h1h2_mask.sum()}")
            print(f"From H2-H3 overlap ({h2h3_start}-{h2h3_end}): {h2h3_mask.sum()}")
            print("\nScore distribution:")
            print(self.gold_standard.describe())

        return self.gold_standard

    def evaluate_model(self,
                       model_col: str,
                       verbose: bool = True) -> Dict:
        """Evaluate model predictions against gold standard"""
        if self.gold_standard is None:
            raise ValueError("Must determine gold standard before evaluating model")

        if verbose:
            print(f"\nEvaluating {model_col} against gold standard...")

        model_scores = pd.to_numeric(self.data[model_col], errors='coerce')

        # Ensure valid scores and alignment
        aligned_data = pd.DataFrame({
            'gold': self.gold_standard,
            'model': model_scores
        })
        aligned_data = aligned_data.dropna()

        if len(aligned_data) == 0:
            raise ValueError(f"No valid comparisons possible between gold standard and {model_col}")

        gold = aligned_data['gold']
        model_scores = aligned_data['model']

        # Calculate metrics
        pearson_r, pearson_p = stats.pearsonr(gold, model_scores)
        spearman_r, spearman_p = stats.spearmanr(gold, model_scores)
        icc = stats.pearsonr(gold, model_scores)[0]
        mse = mean_squared_error(gold, model_scores)
        rmse = np.sqrt(mse)
        agreement_1point = np.mean(np.abs(gold - model_scores) <= 1)

        results = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'icc': icc,
            'mse': mse,
            'rmse': rmse,
            'agreement_1point': agreement_1point,
            'n_samples': len(gold)
        }

        if verbose:
            print("Analysis Results:")
            print("-" * 50)
            print(f"Number of samples analyzed: {results['n_samples']}")
            print(f"Pearson correlation: {results['pearson_r']:.3f} (p={results['pearson_p']:.3f})")
            print(f"Spearman correlation: {results['spearman_r']:.3f} (p={results['spearman_p']:.3f})")
            print(f"ICC: {results['icc']:.3f}")
            print(f"RMSE: {results['rmse']:.3f}")
            print(f"Agreement within ±1 point: {results['agreement_1point']:.3f}")

        return results

    def visualize_results(self, human_results, gpt_results, claude_results):
        """Create visualizations comparing human agreement and model performance"""
        plt.style.use('seaborn')

        # 1. Scatter plots of scores
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Human agreement
        axes[0].scatter(human_results['scores1'], human_results['scores2'], alpha=0.5)
        axes[0].plot([0, 10], [0, 10], 'r--')  # Perfect agreement line
        axes[0].set_xlabel('Human 1 Scores')
        axes[0].set_ylabel('Human 2 Scores')
        axes[0].set_title('Human Agreement\nScatter Plot')

        # GPT vs Gold
        axes[1].scatter(self.gold_standard,
                        pd.to_numeric(self.data['gpt'], errors='coerce'),
                        alpha=0.5)
        axes[1].plot([0, 10], [0, 10], 'r--')
        axes[1].set_xlabel('Gold Standard Scores')
        axes[1].set_ylabel('GPT Scores')
        axes[1].set_title('GPT vs Gold Standard\nScatter Plot')

        # Claude vs Gold
        axes[2].scatter(self.gold_standard,
                        pd.to_numeric(self.data['claude'], errors='coerce'),
                        alpha=0.5)
        axes[2].plot([0, 10], [0, 10], 'r--')
        axes[2].set_xlabel('Gold Standard Scores')
        axes[2].set_ylabel('Claude Scores')
        axes[2].set_title('Claude vs Gold Standard\nScatter Plot')

        plt.tight_layout()
        plt.savefig('score_scatter_plots.png')
        plt.close()

        # 2. Metrics Comparison
        metrics = ['pearson_r', 'spearman_r', 'icc', 'agreement_1point']
        labels = ['Pearson r', 'Spearman r', 'ICC', 'Agreement ±1']

        gpt_scores = [gpt_results[m] for m in metrics]
        claude_scores = [claude_results[m] for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(metrics))
        width = 0.35

        ax.bar([i - width / 2 for i in x], gpt_scores, width, label='GPT', color='skyblue')
        ax.bar([i + width / 2 for i in x], claude_scores, width, label='Claude', color='lightcoral')

        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # Add value labels
        for i, v in enumerate(gpt_scores):
            ax.text(i - width / 2, v, f'{v:.3f}', ha='center', va='bottom')
        for i, v in enumerate(claude_scores):
            ax.text(i + width / 2, v, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('metrics_comparison.png')
        plt.close()


def main():
    # Create analyzer instance
    analyzer = IdeologyScoreAnalyzer()

    # Load data
    analyzer.load_csv('harris_Statement Ideology Score_juxtapose.csv')

    # Define overlap ranges
    overlap_ranges = {
        'h1h2': ('035', '068'),  # replace with actual overlap ranges
        'h2h3': ('069', '102')  # replace with actual overlap ranges
    }

    # Analyze human agreement
    human_results = analyzer.analyze_human_agreement(overlap_ranges= overlap_ranges)

    # Determine gold standard
    gold = analyzer.determine_gold_standard(overlap_ranges=overlap_ranges, strategy='mean')

    # Evaluate models
    gpt_results = analyzer.evaluate_model('gpt')
    claude_results = analyzer.evaluate_model('claude')

    # Create visualizations
    #analyzer.visualize_results(human_results, gpt_results, claude_results)


if __name__ == '__main__':
    main()