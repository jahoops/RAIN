import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

def visualize_ga_results(island_histories, feature_tracking, all_features, homogeneity_tracking=None, 
                        selected_features=None, output_dir="data/visualizations"):
    """Create visualizations of GA population metrics and feature evolution"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert island histories to DataFrame if not already
    if not isinstance(island_histories, pd.DataFrame):
        all_history = []
        for island_idx, history in enumerate(island_histories):
            all_history.extend(history)
        history_df = pd.DataFrame(all_history)
    else:
        history_df = island_histories
        
    print(f"Creating visualizations in {output_dir}...")
    
    # Set up the plots with better styling
    plt.style.use('ggplot')
    
    # 1. Plot average fitness by island
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    for island in range(history_df['island'].max() + 1):
        island_data = history_df[history_df['island'] == island]
        plt.plot(island_data['generation'], island_data['avg_fitness'], 
                label=f'Island {island+1}', linewidth=2)
    plt.title('Average Fitness Score by Generation', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Average Fitness', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. Plot max fitness by island
    plt.subplot(2, 2, 2)
    for island in range(history_df['island'].max() + 1):
        island_data = history_df[history_df['island'] == island]
        plt.plot(island_data['generation'], island_data['max_fitness'], 
                label=f'Island {island+1}', linewidth=2)
    plt.title('Maximum Fitness Score by Generation', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Max Fitness', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 3. Plot homogeneity if available
    plt.subplot(2, 2, 3)
    if homogeneity_tracking is not None:
        homogeneity_df = pd.DataFrame(homogeneity_tracking)
        for island in range(homogeneity_df['island'].max() + 1):
            island_data = homogeneity_df[homogeneity_df['island'] == island]
            plt.plot(island_data['generation'], island_data['homogeneity'], 
                    label=f'Island {island+1}', linewidth=2)
        plt.title('Population Homogeneity by Generation', fontsize=14)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Homogeneity (0=diverse, 1=uniform)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Homogeneity data not available', 
                horizontalalignment='center', verticalalignment='center')
    
    # 4. Plot feature count
    plt.subplot(2, 2, 4)
    for island in range(history_df['island'].max() + 1):
        island_data = history_df[history_df['island'] == island]
        plt.plot(island_data['generation'], island_data['features_used'], 
                label=f'Island {island+1}', linewidth=2)
    plt.title('Number of Features Used by Generation', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Feature Count', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ga_metrics_by_island.png", dpi=300)
    print(f"Saved population metrics chart to {output_dir}/ga_metrics_by_island.png")
    plt.close()
    
    # Feature prevalence heatmap and tracking
    if feature_tracking:
        # Convert to DataFrame if not already
        if not isinstance(feature_tracking, pd.DataFrame):
            feature_prevalence = pd.DataFrame(feature_tracking)
        else:
            feature_prevalence = feature_tracking
            
        # Get unique features and prepare data for tracking
        feature_names = [col for col in feature_prevalence.columns 
                        if col not in ['generation', 'island']]
        
        # For each island, create a features evolution heatmap
        islands = sorted(feature_prevalence['island'].unique())
        
        for island_id in islands:
            # Filter for this island
            island_data = feature_prevalence[feature_prevalence['island'] == island_id].copy()
            
            # Sort by generation to ensure chronological order
            island_data = island_data.sort_values('generation')
            
            # Create matrix for heatmap
            pivot_data = []
            for _, row in island_data.iterrows():
                gen = row['generation']
                for feature in feature_names:
                    pivot_data.append({
                        'generation': gen,
                        'feature': feature,
                        'prevalence': row[feature]
                    })
            
            pivot_df = pd.DataFrame(pivot_data)
            
            # Convert to pivot table
            heatmap_data = pivot_df.pivot(index='feature', columns='generation', values='prevalence')
            
            # Sort features by final prevalence
            if len(heatmap_data.columns) > 0:  # Ensure there's data
                final_gen = max(heatmap_data.columns)
                heatmap_data = heatmap_data.sort_values(by=final_gen, ascending=False)
            
            # Plot top 20 features only for readability
            top_features = 20
            plt.figure(figsize=(15, 12))
            sns.heatmap(heatmap_data.head(top_features), 
                       cmap='viridis', 
                       vmin=0, 
                       vmax=1,
                       xticklabels=10, 
                       cbar_kws={'label': 'Prevalence'})
            plt.title(f'Feature Prevalence Evolution - Island {island_id+1} (Top {top_features} Features)', 
                     fontsize=16)
            plt.xlabel('Generation', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_prevalence_island_{island_id+1}.png", dpi=300)
            print(f"Saved feature prevalence heatmap for Island {island_id+1}")
            plt.close()
        
        # Create line chart of feature prevalence for top features across all islands
        # This gives a clearer view of which features are becoming dominant
        plt.figure(figsize=(15, 10))
        
        # For each feature, calculate average prevalence across all islands
        avg_prevalence = {}
        for feature in feature_names:
            avg_prevalence[feature] = []
            
        generations = sorted(feature_prevalence['generation'].unique())
        
        for gen in generations:
            gen_data = feature_prevalence[feature_prevalence['generation'] == gen]
            for feature in feature_names:
                avg = gen_data[feature].mean()
                avg_prevalence[feature].append(avg)
        
        # Get top features by final prevalence
        final_prevalence = {}
        for feature in feature_names:
            final_prevalence[feature] = avg_prevalence[feature][-1]
        
        # Sort and get top features
        top_n = 15
        top_features = sorted(final_prevalence.keys(), 
                            key=lambda x: final_prevalence[x], reverse=True)[:top_n]
        
        # Plot top features
        for feature in top_features:
            plt.plot(generations, avg_prevalence[feature], 
                    label=feature, linewidth=2)
        
        plt.title(f'Top {top_n} Features by Final Prevalence (All Islands)', fontsize=16)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Average Prevalence', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_features_prevalence.png", dpi=300)
        print(f"Saved top features prevalence chart")
        plt.close()
        
        # If we have selected features, show how they evolved specifically
        if selected_features:
            plt.figure(figsize=(15, 10))
            
            # Plot selected features with bold lines, others in background
            for feature in feature_names:
                data = avg_prevalence[feature]
                if feature in selected_features:
                    plt.plot(generations, data, label=feature, linewidth=2.5)
                else:
                    plt.plot(generations, data, color='gray', alpha=0.1, linewidth=0.5)
            
            plt.title('Evolution of Selected Features vs. All Features', fontsize=16)
            plt.xlabel('Generation', fontsize=14)
            plt.ylabel('Average Prevalence', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/selected_features_evolution.png", dpi=300)
            print(f"Saved selected features evolution chart")
            plt.close()
    
    print(f"All visualizations saved to {output_dir}/")
    return True