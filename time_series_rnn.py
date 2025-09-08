#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.switch_backend('Agg')

class InnovationTimeSeriesAnalyzer:
    def __init__(self):
        self.patent_data = None
        self.scaler = MinMaxScaler()
        
    def prepare_time_series_data(self):
        """Prepare patent time series data for analysis"""
        # Patent time series data (2017-2023)
        patent_time_series = {
            'Year': list(range(2017, 2024)),
            'Beijing': [46091, 46978, 53127, 63266, 79210, 88127, 107875],
            'Shanghai': [20681, 21331, 22735, 24208, 32860, 36800, 44345],
            'Shenzhen': [18926, 21309, 26051, 31138, 45202, 52172, 62252],
            'Guangzhou': [9345, 10797, 12222, 15077, 24120, 27604, 36339],
            'Tianjin': [5844, 5626, 5025, 5262, 7376, 11745, 14319],
            'Wuhan': [8444, 8807, 33202, 34635, 18553, 23658, 22751]
        }
        
        self.patent_data = pd.DataFrame(patent_time_series)
        return self.patent_data
    
    def analyze_growth_patterns(self):
        """Analyze growth patterns and trends"""
        df = self.patent_data.copy()
        
        # Calculate growth rates
        cities = df.columns[1:]  # Exclude 'Year' column
        growth_stats = {}
        
        for city in cities:
            values = df[city].values
            growth_rates = [(values[i] - values[i-1]) / values[i-1] * 100 
                           for i in range(1, len(values))]
            
            growth_stats[city] = {
                'total_growth': ((values[-1] - values[0]) / values[0] * 100),
                'avg_annual_growth': np.mean(growth_rates),
                'growth_volatility': np.std(growth_rates),
                'compound_growth': ((values[-1] / values[0]) ** (1/6) - 1) * 100
            }
        
        # Visualize growth patterns
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Time series plot
        for city in cities:
            ax1.plot(df['Year'], df[city], marker='o', linewidth=2, label=city)
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Patent Authorizations')
        ax1.set_title('Patent Authorization Trends (2017-2023)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Growth rate comparison
        growth_data = pd.DataFrame(growth_stats).T
        growth_data.plot(kind='bar', y='compound_growth', ax=ax2, color='steelblue', alpha=0.7)
        ax2.set_title('Compound Annual Growth Rate (CAGR)', fontweight='bold')
        ax2.set_ylabel('CAGR (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Volatility analysis
        growth_data.plot(kind='scatter', x='avg_annual_growth', y='growth_volatility', 
                        s=100, ax=ax3, alpha=0.7)
        
        for city in growth_data.index:
            ax3.annotate(city, (growth_data.loc[city, 'avg_annual_growth'], 
                              growth_data.loc[city, 'growth_volatility']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Average Annual Growth Rate (%)')
        ax3.set_ylabel('Growth Volatility (Std Dev)')
        ax3.set_title('Growth vs Volatility Matrix', fontweight='bold')
        
        # Logarithmic scale for better trend visualization
        for city in cities:
            ax4.semilogy(df['Year'], df[city], marker='o', linewidth=2, label=city)
        
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Patent Authorizations (log scale)')
        ax4.set_title('Innovation Trends (Log Scale)', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return growth_stats
    
    def simple_lstm_prediction(self):
        """Simple LSTM-style prediction without TensorFlow"""
        df = self.patent_data.copy()
        cities = df.columns[1:]
        predictions = {}
        
        # Simple polynomial fitting and extrapolation
        years = df['Year'].values
        future_years = np.array([2024, 2025, 2026])
        
        plt.figure(figsize=(16, 10))
        
        for i, city in enumerate(cities):
            plt.subplot(2, 3, i+1)
            
            values = df[city].values
            
            # Fit polynomial (degree 2 for growth curve)
            coeffs = np.polyfit(years, values, deg=2)
            poly_func = np.poly1d(coeffs)
            
            # Generate smooth curve for visualization
            extended_years = np.linspace(2017, 2026, 50)
            smooth_curve = poly_func(extended_years)
            
            # Predict future values
            future_predictions = poly_func(future_years)
            predictions[city] = future_predictions
            
            # Plot historical data
            plt.plot(years, values, 'o-', label='Historical', linewidth=2, markersize=6)
            
            # Plot fitted curve
            plt.plot(extended_years, smooth_curve, '--', alpha=0.7, label='Trend')
            
            # Plot predictions
            plt.plot(future_years, future_predictions, 's', 
                    color='red', markersize=8, label='Predicted')
            
            plt.title(f'{city} Patent Predictions', fontweight='bold')
            plt.xlabel('Year')
            plt.ylabel('Patents')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Calculate R² for model fit
            predicted_historical = poly_func(years)
            r2 = 1 - (np.sum((values - predicted_historical) ** 2) / 
                     np.sum((values - np.mean(values)) ** 2))
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('/workspace/patent_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return predictions
    
    def innovation_momentum_analysis(self):
        """Analyze innovation momentum and acceleration"""
        df = self.patent_data.copy()
        cities = df.columns[1:]
        
        momentum_data = {}
        
        for city in cities:
            values = df[city].values
            years = df['Year'].values
            
            # Calculate first derivative (velocity - growth rate)
            velocity = np.gradient(values)
            
            # Calculate second derivative (acceleration - rate of change in growth)
            acceleration = np.gradient(velocity)
            
            momentum_data[city] = {
                'current_velocity': velocity[-1],
                'current_acceleration': acceleration[-1],
                'momentum_score': velocity[-1] + 0.5 * acceleration[-1]
            }
        
        # Create momentum visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Velocity vs Acceleration quadrant analysis
        velocities = [momentum_data[city]['current_velocity'] for city in cities]
        accelerations = [momentum_data[city]['current_acceleration'] for city in cities]
        
        scatter = ax1.scatter(velocities, accelerations, s=100, alpha=0.7, c=range(len(cities)), cmap='viridis')
        
        for i, city in enumerate(cities):
            ax1.annotate(city, (velocities[i], accelerations[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Add quadrant lines
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        ax1.set_xlabel('Innovation Velocity (Current Growth Rate)')
        ax1.set_ylabel('Innovation Acceleration (Growth Change Rate)')
        ax1.set_title('Innovation Momentum Quadrant Analysis', fontweight='bold')
        
        # Momentum scores ranking
        momentum_scores = [momentum_data[city]['momentum_score'] for city in cities]
        
        ax2.barh(cities, momentum_scores, color='coral', alpha=0.7)
        ax2.set_xlabel('Innovation Momentum Score')
        ax2.set_title('City Innovation Momentum Rankings', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/workspace/momentum_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return momentum_data
    
    def run_complete_analysis(self):
        """Run complete time series analysis"""
        print("📈 Starting Time Series & Predictive Analysis...")
        print("=" * 50)
        
        # Prepare data
        self.prepare_time_series_data()
        print("✅ Time series data prepared")
        
        # Growth pattern analysis
        growth_stats = self.analyze_growth_patterns()
        print("✅ Growth pattern analysis complete")
        
        # Future predictions
        predictions = self.simple_lstm_prediction()
        print("✅ Patent predictions generated")
        
        # Momentum analysis
        momentum_data = self.innovation_momentum_analysis()
        print("✅ Innovation momentum analysis complete")
        
        # Print key insights
        print("\n🔍 KEY INSIGHTS:")
        print("=" * 30)
        
        # Top performers by different metrics
        growth_df = pd.DataFrame(growth_stats).T
        
        print("📊 Compound Annual Growth Rate Leaders:")
        top_growth = growth_df.nlargest(3, 'compound_growth')
        for i, (city, data) in enumerate(top_growth.iterrows(), 1):
            print(f"{i}. {city}: {data['compound_growth']:.1f}% CAGR")
        
        print("\n🚀 Innovation Momentum Leaders:")
        momentum_df = pd.DataFrame(momentum_data).T
        top_momentum = momentum_df.nlargest(3, 'momentum_score')
        for i, (city, data) in enumerate(top_momentum.iterrows(), 1):
            print(f"{i}. {city}: {data['momentum_score']:.0f} momentum score")
        
        print("\n🔮 2026 Patent Predictions:")
        for city, pred_values in predictions.items():
            print(f"{city}: {pred_values[-1]:.0f} patents")
        
        print(f"\n📊 Generated visualizations:")
        print("- time_series_analysis.png")
        print("- patent_predictions.png")
        print("- momentum_analysis.png")
        
        return {
            'growth_stats': growth_stats,
            'predictions': predictions,
            'momentum_data': momentum_data
        }

if __name__ == "__main__":
    analyzer = InnovationTimeSeriesAnalyzer()
    results = analyzer.run_complete_analysis()