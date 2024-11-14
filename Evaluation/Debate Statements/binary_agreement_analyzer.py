import pandas as pd
import numpy as np
from typing import Union, Dict
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class BinaryAnalyzer:
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

    def _convert_to_binary(self, scores: pd.Series) -> pd.Series:
        """Convert scores to binary (1 for -1, 0 for others)"""
        return (pd.to_numeric(scores, errors='coerce') == -1).astype(float)

    def analyze_pairwise_human_agreement(self, human1_col, human2_col, id_range=None, description=""):
        """
        Analyze agreement between a pair of human annotators within specified ID range

        Args:
            human1_col: name of first human annotator column
            human2_col: name of second human annotator column
            id_range: tuple of (start_id, end_id) for overlapping region
            description: description of the comparison
        """
        if id_range:
            start_id, end_id = id_range
            # Filter data for specified range
            mask = (self.data.index >= str(start_id).zfill(3)) & (self.data.index <= str(end_id).zfill(3))
            data_slice = self.data[mask]
        else:
            data_slice = self.data

        binary1 = self._convert_to_binary(data_slice[human1_col])
        binary2 = self._convert_to_binary(data_slice[human2_col])

        # Only consider where both annotators have annotations
        valid_mask = ~(binary1.isna() | binary2.isna())
        binary1 = binary1[valid_mask]
        binary2 = binary2[valid_mask]

        if len(binary1) == 0:
            print(f"No overlapping annotations found for {description}")
            return None

        kappa = cohen_kappa_score(binary1, binary2)
        agreement = (binary1 == binary2).mean()
        confusion = pd.crosstab(binary1, binary2)

        print(f"\nAgreement Analysis for {description}:")
        print(f"ID Range: {id_range if id_range else 'all'}")
        print(f"Number of overlapping annotations: {len(binary1)}")
        print(f"Kappa: {kappa:.3f}")
        print(f"Agreement: {agreement:.3f}")
        print("\nConfusion Matrix:")
        print(confusion)

        return {
            'kappa': kappa,
            'agreement': agreement,
            'confusion_matrix': confusion,
            'n_samples': len(binary1)
        }

    def analyze_human_agreements(self, overlap_ranges):
        """
        Analyze agreements between human annotators using specified overlap ranges

        Args:
            overlap_ranges: dict with keys 'h1h2' and 'h2h3' containing tuples of (start_id, end_id)
        """
        results = {}

        # Analyze H1 vs H2 in their overlap range
        results['h1h2'] = self.analyze_pairwise_human_agreement(
            'human1', 'human2',
            id_range=overlap_ranges['h1h2'],
            description='Human 1 vs Human 2'
        )

        # Analyze H2 vs H3 in their overlap range
        results['h2h3'] = self.analyze_pairwise_human_agreement(
            'human2', 'human3',
            id_range=overlap_ranges['h2h3'],
            description='Human 2 vs Human 3'
        )

        return results

    def determine_gold_standard(self, overlap_ranges):
        """
        Determine gold standard using specified overlap ranges

        Args:
            overlap_ranges: dict with keys 'h1h2' and 'h2h3' containing tuples of (start_id, end_id)
        """
        binary1 = self._convert_to_binary(self.data['human1'])
        binary2 = self._convert_to_binary(self.data['human2'])
        binary3 = self._convert_to_binary(self.data['human3'])

        self.gold_standard = pd.Series(index=self.data.index, dtype=float)

        # Handle H1-H2 overlap region
        h1h2_start, h1h2_end = overlap_ranges['h1h2']
        h1h2_mask = (self.data.index >= str(h1h2_start).zfill(3)) & (self.data.index <= str(h1h2_end).zfill(3))
        self.gold_standard[h1h2_mask] = ((binary1[h1h2_mask] + binary2[h1h2_mask]) >= 1).astype(float)

        # Handle H2-H3 overlap region
        h2h3_start, h2h3_end = overlap_ranges['h2h3']
        h2h3_mask = (self.data.index >= str(h2h3_start).zfill(3)) & (self.data.index <= str(h2h3_end).zfill(3))
        self.gold_standard[h2h3_mask] = ((binary2[h2h3_mask] + binary3[h2h3_mask]) >= 1).astype(float)

        # Handle non-overlapping regions with single annotations
        for idx in self.data.index:
            if pd.isna(self.gold_standard[idx]):
                # Check each annotator in priority order
                if not pd.isna(binary1[idx]):
                    self.gold_standard[idx] = binary1[idx]
                elif not pd.isna(binary2[idx]):
                    self.gold_standard[idx] = binary2[idx]
                elif not pd.isna(binary3[idx]):
                    self.gold_standard[idx] = binary3[idx]

        print("\nGold Standard Statistics:")
        print(f"Total annotations: {len(self.gold_standard.dropna())}")
        print(f"From H1-H2 overlap ({h1h2_start}-{h1h2_end}): {h1h2_mask.sum()}")
        print(f"From H2-H3 overlap ({h2h3_start}-{h2h3_end}): {h2h3_mask.sum()}")

        return self.gold_standard

    def evaluate_llm(self, llm_col):
        """
        Evaluate LLM performance against gold standard
        Converts LLM scores to binary where -1 is mapped to 1 (polar) and other scores to 0 (non-polar)
        """
        if self.gold_standard is None:
            raise ValueError("Must determine gold standard first")

        # Convert LLM scores to binary
        llm_binary = self._convert_to_binary(self.data[llm_col])

        # Only evaluate where we have gold standard
        valid_mask = ~self.gold_standard.isna()
        gold_standard = self.gold_standard[valid_mask]
        llm_valid = llm_binary[valid_mask]

        # Calculate metrics
        kappa = cohen_kappa_score(gold_standard, llm_valid)
        accuracy = (llm_valid == gold_standard).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(gold_standard, llm_valid, average='binary')

        # Create confusion matrix
        confusion = pd.crosstab(
            gold_standard,
            llm_valid,
            rownames=['Gold'],
            colnames=['LLM']
        )

        print(f"\nEvaluation Results for {llm_col}:")
        print(f"Number of samples: {len(gold_standard)}")
        print(f"Kappa: {kappa:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print("\nConfusion Matrix:")
        print(confusion)

        return {
            'kappa': kappa,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion,
            'n_samples': len(gold_standard)
        }
    def visualize_annotation_coverage(self):
        """Visualize coverage of annotations from different annotators"""
        fig, ax = plt.subplots(figsize=(15, 5))

        coverage_data = pd.DataFrame({
            'Human 1': ~self.data['human1'].isna(),
            'Human 2': ~self.data['human2'].isna(),
            'Human 3': ~self.data['human3'].isna(),
            'Gold Standard': ~self.gold_standard.isna()
        }, index=self.data.index)  # Use the existing index

        # Create heatmap
        sns.heatmap(coverage_data[['Human 1', 'Human 2', 'Human 3', 'Gold Standard']].T,
                    cmap='Blues',
                    cbar_kws={'label': 'Has Annotation'},
                    yticklabels=['Human 1', 'Human 2', 'Human 3', 'Gold Standard'])

        plt.title('Annotation Coverage by Source')
        plt.xlabel('Statement Index')
        plt.tight_layout()
        plt.savefig('annotation_coverage.png')
        plt.close()


def main():
    file_path = "harris_Statement Ideology Score_juxtapose.csv"
    print(f"Attempting to load file: {file_path}")

    try:
        analyzer = BinaryAnalyzer()
        analyzer.load_csv(file_path)
        # Define overlap ranges
        overlap_ranges = {
            'h1h2': (35, 68),  # example range where human1 and human2 overlap
            'h2h3': (69, 136)  # example range where human2 and human3 overlap
        }

        # Analyze pairwise agreements between human annotators
        print("\nAnalyzing human agreements...")
        human_agreements = analyzer.analyze_human_agreements(overlap_ranges)

        # Determine gold standard
        print("\nDetermining gold standard...")
        analyzer.determine_gold_standard(overlap_ranges)

        # Visualize annotation coverage
        print("\nGenerating annotation coverage visualization...")
        analyzer.visualize_annotation_coverage()

        # Evaluate LLMs against gold standard
        print("\nEvaluating GPT against gold standard...")
        gpt_results = analyzer.evaluate_llm('gpt')
        print(gpt_results)

        print("\nEvaluating Claude against gold standard...")
        claude_results = analyzer.evaluate_llm('claude')
        print(claude_results)

        # Generate comparison visualizations
        #print("\nGenerating performance comparison visualizations...")
        #analyzer.visualize_results(human_agreements, gpt_results, claude_results)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()