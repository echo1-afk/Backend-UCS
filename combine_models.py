#!/usr/bin/env python3
"""
Combine predictions from LSTM and CNN-LSTM models
Run this AFTER both lstm_model.py and cnn_lstm_model.py have completed
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import os

def check_files_exist():
    """Check if required model files exist"""
    required_files = [
        'lstm_model.h5',
        'cnn_lstm_model.h5',
        'lstm_predictions.csv',
        'cnn_lstm_predictions.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\n❌ ERROR: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run both lstm_model.py and cnn_lstm_model.py first!")
        return False
    
    print("✓ All required files found")
    return True

def load_models():
    """Load saved models"""
    print("\n" + "="*70)
    print("Loading Saved Models")
    print("="*70)
    
    lstm_model = tf.keras.models.load_model('lstm_model.h5', compile=False)
    print("✓ Loaded lstm_model.h5")
    
    cnn_lstm_model = tf.keras.models.load_model('cnn_lstm_model.h5', compile=False)
    print("✓ Loaded cnn_lstm_model.h5")
    
    return lstm_model, cnn_lstm_model

def load_predictions():
    """Load saved predictions"""
    print("\n" + "="*70)
    print("Loading Predictions")
    print("="*70)
    
    lstm_preds = pd.read_csv('lstm_predictions.csv')
    print(f"✓ Loaded lstm_predictions.csv - {len(lstm_preds)} samples")
    
    cnn_lstm_preds = pd.read_csv('cnn_lstm_predictions.csv')
    print(f"✓ Loaded cnn_lstm_predictions.csv - {len(cnn_lstm_preds)} samples")
    
    return lstm_preds, cnn_lstm_preds

def evaluate_individual_models(lstm_preds, cnn_lstm_preds):
    """Evaluate each model individually"""
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*70)
    
    # LSTM metrics
    lstm_rmse = np.sqrt(mean_squared_error(lstm_preds['Actual'], lstm_preds['Predicted']))
    lstm_mae = mean_absolute_error(lstm_preds['Actual'], lstm_preds['Predicted'])
    lstm_r2 = r2_score(lstm_preds['Actual'], lstm_preds['Predicted'])
    
    print("\n--- LSTM Model ---")
    print(f"RMSE: {lstm_rmse:.4f}")
    print(f"MAE:  {lstm_mae:.4f}")
    print(f"R²:   {lstm_r2:.4f}")
    
    # CNN-LSTM metrics
    cnn_rmse = np.sqrt(mean_squared_error(cnn_lstm_preds['Actual'], cnn_lstm_preds['Predicted']))
    cnn_mae = mean_absolute_error(cnn_lstm_preds['Actual'], cnn_lstm_preds['Predicted'])
    cnn_r2 = r2_score(cnn_lstm_preds['Actual'], cnn_lstm_preds['Predicted'])
    
    print("\n--- CNN-LSTM Model ---")
    print(f"RMSE: {cnn_rmse:.4f}")
    print(f"MAE:  {cnn_mae:.4f}")
    print(f"R²:   {cnn_r2:.4f}")
    
    return {
        'LSTM': {'RMSE': lstm_rmse, 'MAE': lstm_mae, 'R2': lstm_r2},
        'CNN-LSTM': {'RMSE': cnn_rmse, 'MAE': cnn_mae, 'R2': cnn_r2}
    }

def create_ensemble(lstm_preds, cnn_lstm_preds, individual_metrics):
    """Create ensemble predictions"""
    print("\n" + "="*70)
    print("ENSEMBLE MODEL PERFORMANCE")
    print("="*70)
    
    actual = lstm_preds['Actual'].values
    lstm_pred = lstm_preds['Predicted'].values
    cnn_pred = cnn_lstm_preds['Predicted'].values
    
    # Simple average ensemble
    ensemble_simple = (lstm_pred + cnn_pred) / 2
    
    # Weighted ensemble based on R² scores
    lstm_r2 = individual_metrics['LSTM']['R2']
    cnn_r2 = individual_metrics['CNN-LSTM']['R2']
    total_r2 = lstm_r2 + cnn_r2
    
    lstm_weight = lstm_r2 / total_r2
    cnn_weight = cnn_r2 / total_r2
    
    ensemble_weighted = (lstm_weight * lstm_pred) + (cnn_weight * cnn_pred)
    
    # Evaluate simple ensemble
    simple_rmse = np.sqrt(mean_squared_error(actual, ensemble_simple))
    simple_mae = mean_absolute_error(actual, ensemble_simple)
    simple_r2 = r2_score(actual, ensemble_simple)
    
    print("\n--- Simple Ensemble (50-50 Average) ---")
    print(f"RMSE: {simple_rmse:.4f}")
    print(f"MAE:  {simple_mae:.4f}")
    print(f"R²:   {simple_r2:.4f}")
    
    # Evaluate weighted ensemble
    weighted_rmse = np.sqrt(mean_squared_error(actual, ensemble_weighted))
    weighted_mae = mean_absolute_error(actual, ensemble_weighted)
    weighted_r2 = r2_score(actual, ensemble_weighted)
    
    print(f"\n--- Weighted Ensemble ({lstm_weight:.1%} LSTM, {cnn_weight:.1%} CNN-LSTM) ---")
    print(f"RMSE: {weighted_rmse:.4f}")
    print(f"MAE:  {weighted_mae:.4f}")
    print(f"R²:   {weighted_r2:.4f}")
    
    # Save combined predictions
    combined_df = pd.DataFrame({
        'Actual': actual,
        'LSTM_Predicted': lstm_pred,
        'CNN_LSTM_Predicted': cnn_pred,
        'Ensemble_Simple': ensemble_simple,
        'Ensemble_Weighted': ensemble_weighted
    })
    combined_df.to_csv('combined_predictions.csv', index=False)
    print("\n✓ Saved combined_predictions.csv")
    
    return {
        'Simple': {'RMSE': simple_rmse, 'MAE': simple_mae, 'R2': simple_r2},
        'Weighted': {'RMSE': weighted_rmse, 'MAE': weighted_mae, 'R2': weighted_r2}
    }, combined_df, lstm_weight, cnn_weight

def create_visualizations(combined_df, individual_metrics, ensemble_metrics, lstm_weight, cnn_weight):
    """Create publication-quality visualizations for IEEE paper"""
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)
    
    actual = combined_df['Actual'].values
    lstm_pred = combined_df['LSTM_Predicted'].values
    cnn_pred = combined_df['CNN_LSTM_Predicted'].values
    ensemble_weighted = combined_df['Ensemble_Weighted'].values
    
    # IEEE-friendly colors (grayscale-safe)
    color_actual = '#000000'      # Black
    color_lstm = '#0072BD'        # Blue
    color_cnn = '#D95319'         # Orange-Red
    color_ensemble = '#77AC30'    # Green
    color_perfect = '#A2142F'     # Dark Red
    
    n_samples = min(200, len(actual))
    
    # ========== FIGURE 1: LSTM Model Analysis ==========
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    fig1.patch.set_facecolor('white')
    
    # LSTM - Time series
    ax = axes1[0]
    ax.plot(actual[:n_samples], label='Actual', linewidth=2, color=color_actual, linestyle='-')
    ax.plot(lstm_pred[:n_samples], label='LSTM Predicted', linewidth=1.5, color=color_lstm, linestyle='--')
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Road Occupancy (%)', fontsize=11)
    ax.set_title('(a) LSTM Predictions vs Actual', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # LSTM - Scatter plot
    ax = axes1[1]
    ax.scatter(actual, lstm_pred, alpha=0.5, s=8, color=color_lstm, edgecolors='none')
    min_val, max_val = actual.min(), actual.max()
    ax.plot([min_val, max_val], [min_val, max_val], color=color_perfect, linestyle='--', linewidth=2, label='Perfect Fit')
    ax.set_xlabel('Actual Values', fontsize=11)
    ax.set_ylabel('LSTM Predicted Values', fontsize=11)
    ax.set_title(f'(b) LSTM Correlation (R² = {individual_metrics["LSTM"]["R2"]:.4f})', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # LSTM - Error distribution
    ax = axes1[2]
    lstm_errors = actual - lstm_pred
    ax.hist(lstm_errors, bins=50, color=color_lstm, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax.axvline(0, color=color_perfect, linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'(c) LSTM Error Distribution (MAE = {individual_metrics["LSTM"]["MAE"]:.3f})', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig('figure_lstm_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved figure_lstm_analysis.png")
    plt.close()
    
    # ========== FIGURE 2: CNN-LSTM Model Analysis ==========
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.patch.set_facecolor('white')
    
    # CNN-LSTM - Time series
    ax = axes2[0]
    ax.plot(actual[:n_samples], label='Actual', linewidth=2, color=color_actual, linestyle='-')
    ax.plot(cnn_pred[:n_samples], label='CNN-LSTM Predicted', linewidth=1.5, color=color_cnn, linestyle='--')
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Road Occupancy (%)', fontsize=11)
    ax.set_title('(a) CNN-LSTM Predictions vs Actual', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # CNN-LSTM - Scatter plot
    ax = axes2[1]
    ax.scatter(actual, cnn_pred, alpha=0.5, s=8, color=color_cnn, edgecolors='none')
    ax.plot([min_val, max_val], [min_val, max_val], color=color_perfect, linestyle='--', linewidth=2, label='Perfect Fit')
    ax.set_xlabel('Actual Values', fontsize=11)
    ax.set_ylabel('CNN-LSTM Predicted Values', fontsize=11)
    ax.set_title(f'(b) CNN-LSTM Correlation (R² = {individual_metrics["CNN-LSTM"]["R2"]:.4f})', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # CNN-LSTM - Error distribution
    ax = axes2[2]
    cnn_errors = actual - cnn_pred
    ax.hist(cnn_errors, bins=50, color=color_cnn, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax.axvline(0, color=color_perfect, linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'(c) CNN-LSTM Error Distribution (MAE = {individual_metrics["CNN-LSTM"]["MAE"]:.3f})', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig('figure_cnn_lstm_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved figure_cnn_lstm_analysis.png")
    plt.close()
    
    # ========== FIGURE 3: Combined Comparison ==========
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    fig3.patch.set_facecolor('white')
    
    # Combined - Time series comparison
    ax = axes3[0, 0]
    ax.plot(actual[:n_samples], label='Actual', linewidth=2.5, color=color_actual, linestyle='-', alpha=0.9)
    ax.plot(lstm_pred[:n_samples], label='LSTM', linewidth=1.5, color=color_lstm, linestyle='--', alpha=0.8)
    ax.plot(cnn_pred[:n_samples], label='CNN-LSTM', linewidth=1.5, color=color_cnn, linestyle='-.', alpha=0.8)
    ax.plot(ensemble_weighted[:n_samples], label='Ensemble', linewidth=2, color=color_ensemble, linestyle='-', alpha=0.9)
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Road Occupancy (%)', fontsize=11)
    ax.set_title('(a) Model Predictions Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Combined - Performance metrics bar chart
    ax = axes3[0, 1]
    models = ['LSTM', 'CNN-LSTM', 'Ensemble']
    rmse_values = [
        individual_metrics['LSTM']['RMSE'],
        individual_metrics['CNN-LSTM']['RMSE'],
        ensemble_metrics['Weighted']['RMSE']
    ]
    mae_values = [
        individual_metrics['LSTM']['MAE'],
        individual_metrics['CNN-LSTM']['MAE'],
        ensemble_metrics['Weighted']['MAE']
    ]
    r2_values = [
        individual_metrics['LSTM']['R2'],
        individual_metrics['CNN-LSTM']['R2'],
        ensemble_metrics['Weighted']['R2']
    ]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, rmse_values, width, label='RMSE', color='#4DBEEE', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, mae_values, width, label='MAE', color='#EDB120', edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, r2_values, width, label='R²', color='#77AC30', edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Models', fontsize=11, fontweight='bold')
    ax.set_ylabel('Metric Values', fontsize=11, fontweight='bold')
    ax.set_title('(b) Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Combined - Ensemble scatter plot
    ax = axes3[1, 0]
    ax.scatter(actual, ensemble_weighted, alpha=0.5, s=10, color=color_ensemble, edgecolors='none')
    ax.plot([min_val, max_val], [min_val, max_val], color=color_perfect, linestyle='--', linewidth=2, label='Perfect Fit')
    ax.set_xlabel('Actual Values', fontsize=11)
    ax.set_ylabel('Ensemble Predicted Values', fontsize=11)
    ax.set_title(f'(c) Ensemble Model (R² = {ensemble_metrics["Weighted"]["R2"]:.4f}, RMSE = {ensemble_metrics["Weighted"]["RMSE"]:.3f})', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Combined - Error comparison
    ax = axes3[1, 1]
    ensemble_errors = actual - ensemble_weighted
    
    ax.hist(lstm_errors, bins=40, alpha=0.6, label='LSTM', color=color_lstm, edgecolor='black', linewidth=0.5)
    ax.hist(cnn_errors, bins=40, alpha=0.6, label='CNN-LSTM', color=color_cnn, edgecolor='black', linewidth=0.5)
    ax.hist(ensemble_errors, bins=40, alpha=0.7, label='Ensemble', color=color_ensemble, edgecolor='black', linewidth=0.8)
    ax.axvline(0, color=color_perfect, linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('(d) Error Distribution Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig('figure_combined_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved figure_combined_comparison.png")
    plt.close()
    
    # ========== FIGURE 4: Performance Summary Table (Optional) ==========
    fig4, ax = plt.subplots(figsize=(8, 4))
    fig4.patch.set_facecolor('white')
    ax.axis('off')
    
    table_data = [
        ['Model', 'RMSE', 'MAE', 'R² Score', 'Weight'],
        ['LSTM', f"{individual_metrics['LSTM']['RMSE']:.4f}", 
         f"{individual_metrics['LSTM']['MAE']:.4f}", 
         f"{individual_metrics['LSTM']['R2']:.4f}",
         f"{lstm_weight:.1%}"],
        ['CNN-LSTM', f"{individual_metrics['CNN-LSTM']['RMSE']:.4f}", 
         f"{individual_metrics['CNN-LSTM']['MAE']:.4f}", 
         f"{individual_metrics['CNN-LSTM']['R2']:.4f}",
         f"{cnn_weight:.1%}"],
        ['Ensemble', f"{ensemble_metrics['Weighted']['RMSE']:.4f}", 
         f"{ensemble_metrics['Weighted']['MAE']:.4f}", 
         f"{ensemble_metrics['Weighted']['R2']:.4f}",
         '-']
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.22, 0.18, 0.18, 0.22, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#CCCCCC')
        cell.set_text_props(weight='bold', fontsize=12)
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
    
    # Style data rows
    for i in range(1, 4):
        for j in range(5):
            cell = table[(i, j)]
            cell.set_facecolor('white')
            cell.set_edgecolor('black')
            cell.set_linewidth(1)
            if j == 0:
                cell.set_text_props(weight='bold')
    
    ax.set_title('TABLE I\nPERFORMANCE COMPARISON OF TRAFFIC PREDICTION MODELS', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('table_performance_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved table_performance_summary.png")
    plt.close()
    
    print("\n" + "="*70)
    print("Generated 4 publication-quality figures:")
    print("  1. figure_lstm_analysis.png - LSTM model analysis")
    print("  2. figure_cnn_lstm_analysis.png - CNN-LSTM model analysis")
    print("  3. figure_combined_comparison.png - Combined comparison")
    print("  4. table_performance_summary.png - Performance table")
    print("="*70)

def create_summary_report(individual_metrics, ensemble_metrics, lstm_weight, cnn_weight):
    """Create a text summary report"""
    print("\n" + "="*70)
    print("Creating Summary Report")
    print("="*70)
    
    report = []
    report.append("="*70)
    report.append("TRAFFIC PREDICTION MODELS - COMBINED ANALYSIS REPORT")
    report.append("="*70)
    report.append("")
    
    report.append("INDIVIDUAL MODEL PERFORMANCE")
    report.append("-" * 70)
    report.append("")
    report.append("LSTM Model:")
    report.append(f"  RMSE: {individual_metrics['LSTM']['RMSE']:.4f}")
    report.append(f"  MAE:  {individual_metrics['LSTM']['MAE']:.4f}")
    report.append(f"  R²:   {individual_metrics['LSTM']['R2']:.4f}")
    report.append("")
    report.append("CNN-LSTM Model:")
    report.append(f"  RMSE: {individual_metrics['CNN-LSTM']['RMSE']:.4f}")
    report.append(f"  MAE:  {individual_metrics['CNN-LSTM']['MAE']:.4f}")
    report.append(f"  R²:   {individual_metrics['CNN-LSTM']['R2']:.4f}")
    report.append("")
    
    report.append("ENSEMBLE MODEL PERFORMANCE")
    report.append("-" * 70)
    report.append("")
    report.append("Simple Ensemble (50-50 Average):")
    report.append(f"  RMSE: {ensemble_metrics['Simple']['RMSE']:.4f}")
    report.append(f"  MAE:  {ensemble_metrics['Simple']['MAE']:.4f}")
    report.append(f"  R²:   {ensemble_metrics['Simple']['R2']:.4f}")
    report.append("")
    report.append(f"Weighted Ensemble ({lstm_weight:.1%} LSTM, {cnn_weight:.1%} CNN-LSTM):")
    report.append(f"  RMSE: {ensemble_metrics['Weighted']['RMSE']:.4f}")
    report.append(f"  MAE:  {ensemble_metrics['Weighted']['MAE']:.4f}")
    report.append(f"  R²:   {ensemble_metrics['Weighted']['R2']:.4f}")
    report.append("")
    
    # Determine best model
    all_models = {
        'LSTM': individual_metrics['LSTM']['R2'],
        'CNN-LSTM': individual_metrics['CNN-LSTM']['R2'],
        'Simple Ensemble': ensemble_metrics['Simple']['R2'],
        'Weighted Ensemble': ensemble_metrics['Weighted']['R2']
    }
    best_model = max(all_models, key=all_models.get)
    
    report.append("BEST MODEL")
    report.append("-" * 70)
    report.append(f"Based on R² score: {best_model} (R² = {all_models[best_model]:.4f})")
    report.append("")
    
    report.append("OUTPUT FILES")
    report.append("-" * 70)
    report.append("  • combined_predictions.csv - All predictions in one file")
    report.append("  • combined_model_analysis.png - Comprehensive visualization")
    report.append("  • combined_analysis_report.txt - This report")
    report.append("")
    report.append("="*70)
    
    report_text = "\n".join(report)
    
    with open('combined_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print("✓ Saved combined_analysis_report.txt")
    print("\n" + report_text)

def main():
    print("\n" + "="*70)
    print("COMBINED MODEL ANALYSIS")
    print("="*70)
    print("\nThis script combines predictions from LSTM and CNN-LSTM models")
    print("="*70)
    
    # Check if files exist
    if not check_files_exist():
        return
    
    # Load models
    lstm_model, cnn_lstm_model = load_models()
    
    # Load predictions
    lstm_preds, cnn_lstm_preds = load_predictions()
    
    # Evaluate individual models
    individual_metrics = evaluate_individual_models(lstm_preds, cnn_lstm_preds)
    
    # Create ensemble
    ensemble_metrics, combined_df, lstm_weight, cnn_weight = create_ensemble(
        lstm_preds, cnn_lstm_preds, individual_metrics
    )
    
    # Create visualizations
    create_visualizations(combined_df, individual_metrics, ensemble_metrics, lstm_weight, cnn_weight)
    
    # Create summary report
    create_summary_report(individual_metrics, ensemble_metrics, lstm_weight, cnn_weight)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  • combined_predictions.csv")
    print("  • combined_model_analysis.png")
    print("  • combined_analysis_report.txt")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
