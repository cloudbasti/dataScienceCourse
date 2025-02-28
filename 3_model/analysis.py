import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_results(file_path="../learning_rate_analysis_intermediate.csv"):
    df = pd.read_csv(file_path)
    
    # Overall performance metrics
    analyze_overall_performance(df)
    
    # Product-specific analysis
    analyze_product_performance(df)
    
    # Learning rate impact visualization
    plot_learning_rate_effects(df)
    
    return df

def analyze_overall_performance(df):
    best_lr_idx = df['r2_score'].idxmax()
    best_metrics = df.loc[best_lr_idx]
    
    print(f"Best Overall Performance:")
    print(f"Learning Rate: {best_metrics['learning_rate']:.6f}")
    print(f"R² Score: {best_metrics['r2_score']:.4f}")
    print(f"RMSE: {best_metrics['rmse']:.2f}")
    print(f"Epochs: {best_metrics['epochs_trained']}")

def analyze_product_performance(df):
    best_lr = df.loc[df['r2_score'].idxmax(), 'learning_rate']
    best_row = df[df['learning_rate'] == best_lr]
    
    print("\nProduct Performance at Best Learning Rate:")
    for i in range(1, 7):
        r2 = best_row[f'Product {i}_r2'].values[0]
        rmse = best_row[f'Product {i}_rmse'].values[0]
        print(f"Product {i}: R² = {r2:.4f}, RMSE = {rmse:.2f}")

def plot_learning_rate_effects(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall R² Score
    ax1.semilogx(df['learning_rate'], df['r2_score'])
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Overall R² Score vs Learning Rate')
    ax1.grid(True)
    
    # Overall RMSE
    ax2.semilogx(df['learning_rate'], df['rmse'])
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Overall RMSE vs Learning Rate')
    ax2.grid(True)
    
    # Product-specific R² scores
    product_r2_cols = [col for col in df.columns if 'Product' in col and '_r2' in col]
    for col in product_r2_cols:
        ax3.semilogx(df['learning_rate'], df[col], label=col.replace('_r2', ''))
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('R² Score')
    ax3.set_title('Product-specific R² Scores')
    ax3.legend()
    ax3.grid(True)
    
    # Heatmap of product performance
    product_data = df.loc[df['r2_score'].idxmax(), product_r2_cols].values.reshape(1, -1)
    sns.heatmap(product_data, 
                ax=ax4,
                annot=True, 
                fmt='.3f',
                cmap='RdYlGn',
                xticklabels=[f'P{i}' for i in range(1, 7)],
                yticklabels=['Best LR'])
    ax4.set_title('Product Performance at Best Learning Rate')
    
    plt.tight_layout()
    plt.savefig('3_model/analysis/learning_rate_analysis_detailed.png')
    plt.close()

if __name__ == "__main__":
    results = load_and_analyze_results('3_model/analysis/learning_rate_analysis_final.csv')