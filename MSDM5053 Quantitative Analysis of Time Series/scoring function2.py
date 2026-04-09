import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class VaRCombiner:
    def __init__(self, pred_df, actual_returns, alpha=0.01):
        """
        pred_df: DataFrame, columns are model names, index is date, values are VaR predictions
        actual_returns: Series, index is date, values are actual returns
        alpha: VaR confidence level (default is 0.01 for 99% VaR)
        """
        self.pred_df = pred_df
        self.actual_returns = actual_returns
        self.alpha = alpha

    def scoring_function(self, VaR, ret):
        indicator = (ret <= VaR).astype(int)
        return (self.alpha - indicator) * (ret - VaR)

    def total_score(self, weights):
        combined_VaR = np.dot(self.pred_df.values, weights)
        score = self.scoring_function(combined_VaR, self.actual_returns.values)
        return np.sum(score)

    def minimum_scoring_combination(self):
        n_models = self.pred_df.shape[1]
        init_weights = np.ones(n_models) / n_models

        cons = ({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        })

        bounds = [(0, 1) for _ in range(n_models)]

        result = minimize(self.total_score, init_weights, bounds=bounds, constraints=cons)

        return result.x, np.dot(self.pred_df.values, result.x)

    def relative_scoring_combination(self):
        scores = self.scoring_function(self.pred_df, self.actual_returns.values.reshape(-1, 1))
        total_scores = np.sum(scores, axis=0)

        delta_scores = total_scores - np.min(total_scores)
        weights = np.exp(-0.5 * delta_scores) / np.sum(np.exp(-0.5 * delta_scores))

        combined_VaR = np.dot(self.pred_df.values, weights)
        return weights, combined_VaR

    def plot_combined_VaR(self, combined_VaR, title="Combined VaR vs Actual Returns"):
        plt.figure(figsize=(12, 6))
        plt.plot(self.actual_returns.index, self.actual_returns.values, label="Actual Returns")
        plt.plot(self.actual_returns.index, -combined_VaR, label="Combined VaR")
        plt.axhline(0, color='black', linestyle='--')
        plt.legend()
        plt.title(title)
        plt.show()


# Example usage:
if __name__ == "__main__":
    # 假设我们有以下CSV文件保存的模型VaR（每列一个模型）
    pred_df_AR = pd.read_csv("Data/GARCH-AR.csv", index_col="Date", parse_dates=True)
    pred_df_Const = pd.read_csv("Data/GARCH-Constant-1.csv", index_col="Date", parse_dates=True)
    pred_df = pd.concat([pred_df_AR["^GSPC"], pred_df_Const["^GSPC"]], axis=1, join='inner')
    pred_df.columns = ["GARCH-AR", "GARCH-Const"]
    print(pred_df.head())
    actual_returns = pd.read_csv("Data/GARCH-Constant-0-actual.csv", index_col="Date", parse_dates=True).squeeze()

    combiner = VaRCombiner(pred_df, actual_returns["^GSPC"])

    # Minimum Scoring Combination
    min_weights, min_combined_VaR = combiner.minimum_scoring_combination()
    print("Minimum Scoring Combination Weights:", min_weights)
    combiner.plot_combined_VaR(min_combined_VaR, title="Minimum Scoring Combination")

    # Relative Scoring Combination
    rel_weights, rel_combined_VaR = combiner.relative_scoring_combination()
    print("Relative Scoring Combination Weights:", rel_weights)
    combiner.plot_combined_VaR(rel_combined_VaR, title="Relative Scoring Combination")