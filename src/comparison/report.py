import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Dict, List, Any


def classifier_comparison_report(
    classifications: Dict[str, List[Any]], title: str = "Classifier Comparison Report"
):
    """
    Generates a statistical and visual comparison of multiple classification methods.

    Parameters:
        classifications (dict): Mapping of method name -> list of classification results.
        title (str): Title for the overall report.
    """
    summary_stats = {}
    cluster_sizes_all = {}

    # Compute summary statistics per classifier
    for name, results in classifications.items():
        if len(results) == 0:
            summary_stats[name] = {"num_groups": 0}
            cluster_sizes_all[name] = []
            continue

        # Handle different result formats
        sizes = []
        if hasattr(results[0], "headlines"):
            # Format: objects with headlines attribute (list of headlines)
            sizes = np.array([len(g.headlines) for g in results])
        elif hasattr(results[0], "entries"):
            # Format: objects with entries attribute (list of entries)
            sizes = np.array([len(g.entries) for g in results])
        elif isinstance(results[0], list):
            # Format: list of lists
            sizes = np.array([len(g) for g in results])
        else:
            # Format: individual items (each result is a single item)
            sizes = np.array([1] * len(results))

        if len(sizes) > 0:
            summary_stats[name] = {
                "num_groups": len(sizes),
                "total_items": int(sizes.sum()),
                "mean_size": float(sizes.mean()),
                "median_size": float(np.median(sizes)),
                "std_size": float(sizes.std()),
                "min_size": int(sizes.min()),
                "max_size": int(sizes.max()),
                "size_entropy": float(entropy(sizes / sizes.sum() + 1e-10)),  # Add small epsilon to avoid log(0)
            }
            cluster_sizes_all[name] = sizes
        else:
            summary_stats[name] = {"num_groups": 0}
            cluster_sizes_all[name] = []

    # Display summary statistics table
    stats_df = pd.DataFrame(summary_stats).T
    print(f"\n{title}")
    print("=" * len(title))
    print(stats_df.round(3))
    print()

    # Print detailed comparison
    print("Detailed Comparison:")
    print("-" * 50)
    for name, stats in summary_stats.items():
        if stats.get("num_groups", 0) > 0:
            print(f"\n{name}:")
            print(f"  Total groups: {stats['num_groups']}")
            print(f"  Total items: {stats.get('total_items', 'N/A')}")
            print(f"  Avg group size: {stats.get('mean_size', 0):.1f}")
            print(f"  Size range: {stats.get('min_size', 0)} - {stats.get('max_size', 0)}")
            
            # Show top 5 largest groups
            if name in cluster_sizes_all and len(cluster_sizes_all[name]) > 0:
                top_sizes = sorted(cluster_sizes_all[name], reverse=True)[:5]
                print(f"  Largest groups: {top_sizes}")

    # Only create plots if we have data with size variations
    valid_data = {name: sizes for name, sizes in cluster_sizes_all.items() 
                  if len(sizes) > 0 and len(np.unique(sizes)) > 1}
    
    if not valid_data:
        print("\nNo data available for plotting (all methods have uniform cluster sizes).")
        return

    # --- Plot 1: Cluster size distribution (KDE) ---
    plt.figure(figsize=(14, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(valid_data)))
    
    for i, (name, sizes) in enumerate(valid_data.items()):
        if len(sizes) > 1:  # Need at least 2 points for KDE
            try:
                sns.kdeplot(sizes, label=name, fill=True, alpha=0.3, color=colors[i])
            except:
                # Fallback to histogram if KDE fails
                plt.hist(sizes, alpha=0.3, label=name, bins=min(10, len(np.unique(sizes))))
    
    plt.xlabel("Cluster size (# headlines)")
    plt.ylabel("Density")
    plt.title("Cluster Size Distributions Across Classifiers")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Boxplot of cluster sizes ---
    if len(valid_data) > 0:
        plt.figure(figsize=(12, 6))
        
        # Prepare data for boxplot
        box_data = []
        box_labels = []
        for name, sizes in valid_data.items():
            box_data.extend(sizes)
            box_labels.extend([name] * len(sizes))
        
        df_box = pd.DataFrame({'Method': box_labels, 'Cluster Size': box_data})
        sns.boxplot(data=df_box, x='Method', y='Cluster Size')
        plt.xticks(rotation=45)
        plt.ylabel("Cluster size (# headlines)")
        plt.title("Comparison of Cluster Sizes Across Classifiers")
        plt.tight_layout()
        plt.show()

    # --- Plot 3: Number of clusters per classifier ---
    num_groups = {name: len(sizes) for name, sizes in cluster_sizes_all.items()}
    plt.figure(figsize=(10, 5))
    bars = plt.bar(list(num_groups.keys()), list(num_groups.values()))
    plt.xticks(rotation=45)
    plt.ylabel("Number of clusters")
    plt.title("Number of Clusters per Classification Method")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def print_classification_samples(classifications: Dict[str, List[Any]], max_samples: int = 3):
    """
    Print sample groups from each classification method for qualitative comparison.
    
    Parameters:
        classifications (dict): Mapping of method name -> list of classification results.
        max_samples (int): Maximum number of sample groups to show per method.
    """
    print("\nSample Groups from Each Classification Method:")
    print("=" * 60)
    
    for name, results in classifications.items():
        print(f"\n{name} - Sample Groups:")
        print("-" * 40)
        
        if len(results) == 0:
            print("  No groups found.")
            continue
            
        sample_count = min(max_samples, len(results))
        
        for i, group in enumerate(results[:sample_count]):
            # Handle different result formats
            items = []
            group_size = 0
            
            if hasattr(group, "headlines"):
                items = [h.title if hasattr(h, 'title') else str(h) for h in group.headlines[:3]]
                group_size = len(group.headlines)
            elif hasattr(group, "entries"):
                items = [e.title if hasattr(e, 'title') else str(e) for e in group.entries[:3]]
                group_size = len(group.entries)
            elif hasattr(group, "headline_text"):
                items = [group.headline_text]
                group_size = len(group.entries) if hasattr(group, "entries") else 1
            elif isinstance(group, list):
                items = [str(item)[:80] + "..." if len(str(item)) > 80 else str(item) for item in group[:3]]
                group_size = len(group)
            else:
                items = [str(group)[:80] + "..." if len(str(group)) > 80 else str(group)]
                group_size = 1
            
            print(f"  Group {i+1} ({group_size} items):")
            for item in items:
                print(f"    • {item}")
            if group_size > 3:
                print(f"    ... and {group_size - 3} more items")
            print()
