import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict, Any
import numpy as np


def load_hero_data(filename: str = "dota2_hero_stats.json") -> Dict[str, Any]:
    """Load hero data from a JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        hero_data = json.load(f)
    return hero_data


def analyze_hero_stats(hero_data: Dict[str, Any]):
    """Perform analysis on hero stats data"""
    # Convert to DataFrame for easier analysis
    heroes_list = list(hero_data.values())
    df = pd.DataFrame(heroes_list)

    print(f"Total heroes: {len(df)}")

    # Basic statistics for numeric columns
    numeric_cols = ['base_health', 'base_mana', 'base_armor', 'base_attack_min',
                    'base_attack_max', 'base_str', 'base_agi', 'base_int',
                    'str_gain', 'agi_gain', 'int_gain', 'attack_range',
                    'projectile_speed', 'move_speed', 'turn_rate']

    print("\nBase stat averages:")
    for col in numeric_cols:
        if col in df.columns:
            avg = df[col].mean()
            print(f"  Average {col}: {avg:.2f}")

    # Attribute type distribution
    print("\nPrimary attribute distribution:")
    attr_counts = df['primary_attr'].value_counts()
    for attr, count in attr_counts.items():
        print(f"  {attr}: {count} heroes")

    # Attack type distribution
    if 'attack_type' in df.columns:
        print("\nAttack type distribution:")
        attack_counts = df['attack_type'].value_counts()
        for attack, count in attack_counts.items():
            print(f"  {attack}: {count} heroes")

    # Roles analysis (if available)
    if 'roles' in df.columns:
        # Convert string roles back to lists if they were joined
        if df['roles'].dtype == 'object' and isinstance(df['roles'].iloc[0], str):
            df['roles'] = df['roles'].apply(lambda x: x.split(','))

        # Count occurrences of each role
        role_counts = {}
        for roles in df['roles']:
            for role in roles:
                role_counts[role] = role_counts.get(role, 0) + 1

        print("\nRole distribution:")
        for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {role}: {count} heroes")

    return df


def plot_hero_stats_matplotlib(df: pd.DataFrame):
    """Create visualizations of hero statistics using matplotlib only"""
    # Set figure size and style
    plt.rcParams['figure.figsize'] = (15, 12)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # Create a figure with multiple subplots
    fig = plt.figure()

    # 1. Primary attribute distribution
    ax1 = fig.add_subplot(2, 2, 1)
    if 'primary_attr' in df.columns:
        attr_counts = df['primary_attr'].value_counts()
        ax1.bar(attr_counts.index, attr_counts.values, color=['red', 'green', 'blue'])
        ax1.set_title('Heroes by Primary Attribute')
        ax1.set_xlabel('Primary Attribute')
        ax1.set_ylabel('Number of Heroes')

    # 2. Base attributes comparison
    ax2 = fig.add_subplot(2, 2, 2)
    attr_cols = ['base_str', 'base_agi', 'base_int']
    if all(col in df.columns for col in attr_cols):
        attr_data = df[attr_cols].mean()
        ax2.bar(attr_data.index, attr_data.values, color=['red', 'green', 'blue'])
        ax2.set_title('Average Base Attributes')
        ax2.set_ylabel('Value')

    # 3. Attribute gain comparison
    ax3 = fig.add_subplot(2, 2, 3)
    gain_cols = ['str_gain', 'agi_gain', 'int_gain']
    if all(col in df.columns for col in gain_cols):
        gain_data = df[gain_cols].mean()
        ax3.bar(gain_data.index, gain_data.values, color=['red', 'green', 'blue'])
        ax3.set_title('Average Attribute Gain per Level')
        ax3.set_ylabel('Value')

    # 4. Attack range distribution
    ax4 = fig.add_subplot(2, 2, 4)
    if 'attack_range' in df.columns:
        # Create histogram
        bins = np.linspace(df['attack_range'].min(), df['attack_range'].max(), 10)
        ax4.hist(df['attack_range'], bins=bins, alpha=0.7, color='purple')
        ax4.set_title('Distribution of Attack Ranges')
        ax4.set_xlabel('Attack Range')
        ax4.set_ylabel('Number of Heroes')

    plt.tight_layout()
    plt.savefig('hero_stats_analysis.png')
    print("Visualizations saved to 'hero_stats_analysis.png'")

    # Create a scatterplot of base stats by attribute
    if 'primary_attr' in df.columns and all(col in df.columns for col in ['base_health', 'base_mana']):
        plt.figure()

        # Get different colors for each attribute type
        colors = {'str': 'red', 'agi': 'green', 'int': 'blue', 'all': 'purple'}

        # Plot each attribute group with different colors
        for attr in df['primary_attr'].unique():
            attr_df = df[df['primary_attr'] == attr]
            plt.scatter(
                attr_df['base_health'],
                attr_df['base_mana'],
                c=colors.get(attr, 'gray'),
                label=attr,
                alpha=0.7,
                s=80  # Point size
            )

        plt.title('Hero Base Health vs Mana by Primary Attribute')
        plt.xlabel('Base Health')
        plt.ylabel('Base Mana')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('hero_stats_scatter.png')
        print("Scatter plot saved to 'hero_stats_scatter.png'")


def main():
    try:
        # Load the hero data
        hero_data = load_hero_data()

        # Analyze the data
        df = analyze_hero_stats(hero_data)

        # Create visualizations with matplotlib
        plot_hero_stats_matplotlib(df)

        print("\nAnalysis completed successfully!")

    except FileNotFoundError:
        print("Hero stats file not found. Please run the data collection script first.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()