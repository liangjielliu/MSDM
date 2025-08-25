import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class VaRCombiner:
    def __init__(self, var_file_paths, actual_returns_path, alpha=0.01):
        """
        var_file_paths: list of file paths to VaR prediction files (CSV).
        actual_returns_path: file path to the actual returns data (CSV).
        alpha: confidence level for VaR (default 0.01 for 99%).
        """
        self.var_file_paths = var_file_paths
        self.actual_returns = self.load_actual_returns(actual_returns_path)
        self.pred_df = self.load_var_files(var_file_paths)
        self.alpha = alpha

    def load_var_files(self, file_paths):
        """
        Load multiple VaR prediction files and combine them into a single DataFrame.
        """
        data_frames = []
        for file_path in file_paths:
            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            data_frames.append(df)
        return pd.concat(data_frames, axis=1)

    def load_actual_returns(self, file_path):
        """
        Load the actual returns data from CSV.
        """
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        return df

    def scoring_function(self, P_t, r_t, alpha=0.01):
        """
        Scoring function as defined in the paper.
        """
        indicator = (r_t <= P_t).astype(int)  # If actual return <= VaR, set indicator to 1
        return (alpha - indicator) * (r_t - P_t)

    def process_asset(self, weights):
        """
        Process each asset's VaR and calculate the score.
        """
        asset_scores = {}

        for i in range(self.pred_df.shape[1]):
            # Get current asset's VaR prediction and actual returns
            asset_var = self.pred_df.iloc[:, i].values
            asset_returns = self.actual_returns.iloc[:, i].values

            # Calculate the weighted VaR
            combined_var = np.dot(asset_var, weights)

            # Calculate the score for the current asset
            score = self.scoring_function(combined_var, asset_returns)

            # Store the asset score
            asset_scores[self.pred_df.columns[i]] = np.sum(score)

            # Output the individual asset score
            print(f"Asset {self.pred_df.columns[i]} score: {np.sum(score)}")

        return asset_scores

    def total_score(self, weights):
        """
        Calculate the total scoring function for the entire portfolio.
        """
        # Process each asset and calculate the total score
        asset_scores = self.process_asset(weights)
        total_score = np.sum(list(asset_scores.values()))
        return total_score

    def minimum_scoring_combination(self):
        """
        Calculate the optimal combination weights by minimizing the scoring function.
        """
        n_models = self.pred_df.shape[1]
        init_weights = np.ones(n_models) / n_models

        cons = ({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1  # Ensure weights sum to 1
        })

        bounds = [(0, 1) for _ in range(n_models)]

        result = minimize(self.total_score, init_weights, bounds=bounds, constraints=cons)

        return result.x, np.dot(self.pred_df.values, result.x)

    def relative_scoring_combination(self):
        """
        Calculate the optimal combination weights using relative scoring.
        """
        scores = self.scoring_function(self.pred_df.values, self.actual_returns.values)
        total_scores = np.sum(scores, axis=0)

        delta_scores = total_scores - np.min(total_scores)
        weights = np.exp(-0.5 * delta_scores) / np.sum(np.exp(-0.5 * delta_scores))

        combined_VaR = np.dot(self.pred_df.values, weights)
        return weights, combined_VaR

    def plot_combined_var(self, combined_VaR, title="Combined VaR vs Actual Returns"):
        """
        Plot the combined VaR and actual returns.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.actual_returns.index, self.actual_returns.values, label="Actual Returns", color='blue')
        plt.plot(self.actual_returns.index, -combined_VaR, label="Combined VaR", color='red', linestyle='--')
        plt.axhline(0, color='black', linestyle='--')
        plt.legend()
        plt.title(title)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Example files - replace with your actual file paths
    var_file_paths = ["Data/GARCH-AR.csv", "Data/GARCH-Const.csv"]  # Paths to your predicted VaR files
    actual_returns_path = "Data/GARCH-Constant-0-actual.csv"  # Path to the actual returns file

    # Create VaRCombiner object
    combiner = VaRCombiner(var_file_paths, actual_returns_path)

    # Calculate the optimal combination of VaR using Minimum Scoring Combination
    weights, combined_VaR = combiner.minimum_scoring_combination()
    print("Optimal Weights:", weights)

    # Plot the combined VaR vs Actual Returns
    combiner.plot_combined_var(combined_VaR, title="Minimum Scoring Combination")

    # Alternatively, calculate the optimal combination using Relative Scoring Combination
    rel_weights, rel_combined_VaR = combiner.relative_scoring_combination()
    print("Relative Scoring Weights:", rel_weights)

    # Plot the combined VaR vs Actual Returns
    combiner.plot_combined_var(rel_combined_VaR, title="Relative Scoring Combination")