import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from typing import Dict, List, Tuple, Union

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataAnalyzer with a pandas DataFrame.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame with columns:
            - Umsatz (turnover)
            - Bewoelkung (cloud cover)
            - Temperatur (temperature)
            - Windgeschwindigkeit (wind speed)
        """
        self.df = df
        self.target_columns = ['Umsatz', 'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit']
        # Validate columns exist
        missing_cols = [col for col in self.target_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
    def get_basic_stats(self) -> Dict:
        """
        Calculate basic statistics for target columns.
        
        Returns:
        Dict: Dictionary containing basic statistics for each column
        """
        stats_dict = {}
        for col in self.target_columns:
            data = self.df[col].dropna()
            stats_dict[col] = {
                'count': len(data),
                'missing_count': self.df[col].isna().sum(),
                'missing_percentage': (self.df[col].isna().sum() / len(self.df)) * 100,
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                '25th': data.quantile(0.25),
                '75th': data.quantile(0.75)
            }
        return stats_dict

    def detect_outliers(self, method: str = 'all', z_threshold: float = 3.0, 
                       iqr_multiplier: float = 1.5) -> Dict:
        """
        Detect outliers using multiple methods.
        
        Parameters:
        method (str): 'zscore', 'iqr', 'modified_zscore', or 'all'
        z_threshold (float): Number of standard deviations for z-score method
        iqr_multiplier (float): IQR multiplier for IQR method
        
        Returns:
        Dict: Dictionary containing outlier indices and summary for each method and column
        """
        outliers_dict = {}
        
        for col in self.target_columns:
            outliers_dict[col] = {}
            data = self.df[col].dropna()
            
            if method in ['zscore', 'all']:
                z_scores = np.abs(stats.zscore(data))
                outliers = data.index[z_scores > z_threshold].tolist()
                outliers_dict[col]['zscore'] = {
                    'indices': outliers,
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(data)) * 100,
                    'values': data[outliers].tolist()
                }
            
            if method in ['iqr', 'all']:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = data.index[
                    (data < (Q1 - iqr_multiplier * IQR)) | 
                    (data > (Q3 + iqr_multiplier * IQR))
                ].tolist()
                outliers_dict[col]['iqr'] = {
                    'indices': outliers,
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(data)) * 100,
                    'values': data[outliers].tolist()
                }
            
            if method in ['modified_zscore', 'all']:
                median = data.median()
                mad = stats.median_abs_deviation(data)
                modified_z_scores = 0.6745 * (data - median) / mad
                outliers = data.index[
                    np.abs(modified_z_scores) > z_threshold
                ].tolist()
                outliers_dict[col]['modified_zscore'] = {
                    'indices': outliers,
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(data)) * 100,
                    'values': data[outliers].tolist()
                }
        
        return outliers_dict
    
    def plot_distributions(self, save_path: str = None):
        """
        Create distribution plots for target columns.
        
        Parameters:
        save_path (str): Path to save the plots (optional)
        """
        fig, axes = plt.subplots(len(self.target_columns), 2, figsize=(15, 5*len(self.target_columns)))
        
        for idx, col in enumerate(self.target_columns):
            # Histogram with KDE
            sns.histplot(data=self.df, x=col, kde=True, ax=axes[idx, 0])
            axes[idx, 0].set_title(f'{col} Distribution')
            
            # Box plot
            sns.boxplot(data=self.df, y=col, ax=axes[idx, 1])
            axes[idx, 1].set_title(f'{col} Box Plot')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def analyze_relationships(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze relationships between target columns.
        
        Returns:
        Tuple containing:
        - Correlation matrix
        - Dictionary with additional relationship metrics
        """
        # Create clean dataframe without missing values
        clean_df = self.df[self.target_columns].dropna()
        
        # Correlation analysis using different methods
        pearson_corr = clean_df.corr(method='pearson')
        spearman_corr = clean_df.corr(method='spearman')
        kendall_corr = clean_df.corr(method='kendall')
        
        # Additional relationship metrics
        relationships = {}
        
        # Analyze relationship between weather variables and turnover
        for col in ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit']:
            # Calculate different correlation coefficients with Umsatz
            clean_data = pd.concat([self.df['Umsatz'], self.df[col]], axis=1).dropna()
            
            pearson = stats.pearsonr(clean_data['Umsatz'], clean_data[col])
            spearman = stats.spearmanr(clean_data['Umsatz'], clean_data[col])
            kendall = stats.kendalltau(clean_data['Umsatz'], clean_data[col])
            
            # Perform additional statistical tests
            # Check for monotonic relationship
            f_stat, f_pval = stats.f_oneway(clean_data['Umsatz'], clean_data[col])
            
            relationships[col] = {
                'pearson_correlation': pearson.statistic,
                'pearson_pvalue': pearson.pvalue,
                'spearman_correlation': spearman.correlation,
                'spearman_pvalue': spearman.pvalue,
                'kendall_correlation': kendall.correlation,
                'kendall_pvalue': kendall.pvalue,
                'f_statistic': f_stat,
                'f_pvalue': f_pval,
                'sample_size': len(clean_data)
            }
        
        return pearson_corr, relationships
    
    def suggest_outlier_treatment(self) -> Dict:
        """
        Analyze outlier patterns and suggest treatment methods.
        
        Returns:
        Dict: Dictionary with suggestions for each column
        """
        suggestions = {}
        
        for col in self.target_columns:
            data = self.df[col].dropna()
            
            # Calculate various metrics to inform suggestions
            skewness = abs(data.skew())
            kurtosis = data.kurtosis()
            outliers_iqr = self.detect_outliers(method='iqr')[col]['iqr']
            outliers_zscore = self.detect_outliers(method='zscore')[col]['zscore']
            
            suggestion = {
                'distribution_type': 'normal' if abs(skewness) < 0.5 else 'skewed',
                'outlier_percentage_iqr': outliers_iqr['percentage'],
                'outlier_percentage_zscore': outliers_zscore['percentage']
            }
            
            # Make suggestions based on the analysis
            if col == 'Umsatz':
                if skewness > 1:
                    suggestion['methods'] = [
                        'Consider log transformation before outlier detection',
                        'Use robust scaling methods',
                        'Consider modified z-score method for outlier detection'
                    ]
                else:
                    suggestion['methods'] = [
                        'Standard z-score or IQR methods should work well',
                        'Consider mean/median based on distribution shape'
                    ]
            else:  # Weather variables
                suggestion['methods'] = [
                    'Weather variables might have natural extremes',
                    'Consider domain knowledge for outlier thresholds',
                    'Validate extreme values against historical weather data'
                ]
            
            suggestions[col] = suggestion
            
        return suggestions

# Example usage:
def main():
    # Set up the correct path
    import os
    
    # Get the current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Construct path to the data file
    data_path = os.path.join(current_dir, 'data', 'merged_data.csv')
    print(f"Attempting to read file from: {data_path}")
    
    # Load the data
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: Could not find file at {data_path}")
        print("Please ensure you're running the script from the data_science directory")
        return
    
    # Initialize analyzer
    analyzer = DataAnalyzer(df)
    
    # Get basic statistics
    basic_stats = analyzer.get_basic_stats()
    print("Basic Statistics:")
    print(pd.DataFrame(basic_stats))
    
    # Detect outliers using all methods
    outliers = analyzer.detect_outliers()
    print("\nOutlier Detection Results:")
    for col in outliers:
        print(f"\n{col}:")
        for method, result in outliers[col].items():
            print(f"{method}: {result['count']} outliers detected ({result['percentage']:.2f}%)")
    
    # Create distribution plots
    analyzer.plot_distributions(save_path='distributions.png')
    
    # Analyze relationships
    corr_matrix, relationships = analyzer.analyze_relationships()
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    print("\nDetailed Relationships:")
    print(pd.DataFrame(relationships))
    
    # Get treatment suggestions
    suggestions = analyzer.suggest_outlier_treatment()
    print("\nOutlier Treatment Suggestions:")
    print(pd.DataFrame(suggestions))

if __name__ == "__main__":
    main()