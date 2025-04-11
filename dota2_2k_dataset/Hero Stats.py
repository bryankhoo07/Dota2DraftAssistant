import requests
import json
import pandas as pd
import time
from typing import List, Dict, Any, Optional


class Dota2HeroStats:
    """Class to retrieve and process Dota 2 hero statistics from OpenDota API"""

    def __init__(self):
        # Base API URLs for OpenDota
        self.base_url = "https://api.opendota.com/api"
        self.heroes_url = f"{self.base_url}/heroes"
        self.hero_stats_url = f"{self.base_url}/heroStats"

        # Headers for API requests
        self.headers = {
            "User-Agent": "Dota2HeroStatsScript/1.0",
            "Accept": "application/json"
        }

        # Rate limiting to avoid API restrictions
        self.request_delay = 1  # seconds between requests

    def get_all_heroes(self) -> List[Dict[str, Any]]:
        """Retrieve basic information about all heroes"""
        print("Fetching list of all heroes...")
        response = requests.get(self.heroes_url, headers=self.headers)

        if response.status_code == 200:
            heroes = response.json()
            print(f"Successfully retrieved {len(heroes)} heroes")
            return heroes
        else:
            print(f"Error retrieving heroes: {response.status_code}")
            print(response.text)
            return []

    def get_detailed_hero_stats(self) -> List[Dict[str, Any]]:
        """Retrieve detailed statistics for all heroes"""
        print("Fetching detailed hero statistics...")
        response = requests.get(self.hero_stats_url, headers=self.headers)

        if response.status_code == 200:
            hero_stats = response.json()
            print(f"Successfully retrieved detailed stats for {len(hero_stats)} heroes")
            return hero_stats
        else:
            print(f"Error retrieving hero stats: {response.status_code}")
            print(response.text)
            return []

    def get_hero_matchups(self, hero_id: int) -> List[Dict[str, Any]]:
        """Retrieve matchup data for a specific hero"""
        matchups_url = f"{self.base_url}/heroes/{hero_id}/matchups"
        print(f"Fetching matchups for hero ID {hero_id}...")

        response = requests.get(matchups_url, headers=self.headers)

        if response.status_code == 200:
            matchups = response.json()
            print(f"Successfully retrieved {len(matchups)} matchups for hero ID {hero_id}")
            return matchups
        else:
            print(f"Error retrieving matchups for hero ID {hero_id}: {response.status_code}")
            print(response.text)
            return []

    def get_hero_recent_performance(self, hero_id: int) -> Dict[str, Any]:
        """Retrieve recent performance data for a specific hero"""
        performance_url = f"{self.base_url}/heroes/{hero_id}/performance"
        print(f"Fetching performance data for hero ID {hero_id}...")

        response = requests.get(performance_url, headers=self.headers)

        if response.status_code == 200:
            performance = response.json()
            print(f"Successfully retrieved performance data for hero ID {hero_id}")
            return performance
        else:
            print(f"Error retrieving performance for hero ID {hero_id}: {response.status_code}")
            print(response.text)
            return {}

    def collect_all_hero_data(self, include_matchups: bool = False) -> Dict[str, Any]:
        """
        Collect comprehensive data for all heroes

        Args:
            include_matchups: Whether to include hero matchup data (can be slow)

        Returns:
            Dictionary with complete hero data
        """
        # Get basic hero info
        heroes = self.get_all_heroes()
        time.sleep(self.request_delay)

        # Get detailed hero stats
        hero_stats = self.get_detailed_hero_stats()
        time.sleep(self.request_delay)

        # Combine the data
        hero_data = {}
        for hero in heroes:
            hero_id = hero['id']
            hero_data[hero_id] = {
                'id': hero_id,
                'name': hero['name'],
                'localized_name': hero['localized_name'],
                'primary_attr': hero.get('primary_attr', ''),
                'attack_type': hero.get('attack_type', ''),
                'roles': hero.get('roles', []),
            }

        # Add detailed stats
        for stat in hero_stats:
            hero_id = stat['id']
            if hero_id in hero_data:
                # Add base stats
                for key, value in stat.items():
                    if key not in hero_data[hero_id]:
                        hero_data[hero_id][key] = value

        # Add matchups if requested (this can be slow)
        if include_matchups:
            for hero_id in hero_data:
                print(f"Processing hero ID {hero_id}: {hero_data[hero_id]['localized_name']}")
                matchups = self.get_hero_matchups(hero_id)
                hero_data[hero_id]['matchups'] = matchups
                time.sleep(self.request_delay)

        return hero_data

    def save_hero_data(self, hero_data: Dict[str, Any], filename: str = "dota2_hero_stats.json"):
        """Save hero data to a JSON file"""
        print(f"Saving hero data to {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(hero_data, f, indent=2)
        print(f"Data saved successfully to {filename}")

    def export_to_csv(self, hero_data: Dict[str, Any], filename: str = "dota2_hero_stats.csv"):
        """Export hero data to a CSV file"""
        print(f"Exporting hero data to {filename}...")

        # Convert to DataFrame
        heroes_list = list(hero_data.values())

        # Normalize the roles column which is a list
        for hero in heroes_list:
            if 'roles' in hero and isinstance(hero['roles'], list):
                hero['roles'] = ','.join(hero['roles'])

        df = pd.DataFrame(heroes_list)

        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Data exported successfully to {filename}")


def main():
    # Create the hero stats retriever
    dota_stats = Dota2HeroStats()

    # Collect all hero data (set include_matchups=True if you want matchup data too)
    hero_data = dota_stats.collect_all_hero_data(include_matchups=False)

    # Save the data
    dota_stats.save_hero_data(hero_data, "dota2_hero_stats.json")
    dota_stats.export_to_csv(hero_data, "dota2_hero_stats.csv")

    # Display summary
    print("\nHero Stats Summary:")
    print(f"Total heroes: {len(hero_data)}")

    # Group heroes by primary attribute
    attr_counts = {}
    for hero_id, hero in hero_data.items():
        attr = hero.get('primary_attr', 'unknown')
        attr_counts[attr] = attr_counts.get(attr, 0) + 1

    print("\nHeroes by primary attribute:")
    for attr, count in attr_counts.items():
        print(f"  {attr}: {count}")


if __name__ == "__main__":
    main()