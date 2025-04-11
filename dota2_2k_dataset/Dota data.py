import requests
import pandas as pd
import os
import json
import time
from datetime import datetime, timedelta
import pickle


class DotaHeroMetadata:
    def __init__(self, heroes_file=None):
        # Default hero attributes (strength, agility, intelligence)
        self.hero_attributes = {}
        # Default hero roles (position 1-5, or carry, mid, offlane, soft support, hard support)
        self.hero_roles = {}
        # Hero pick rates and win rates (to be populated)
        self.hero_pick_rates = {}
        self.hero_win_rates = {}

        # Load hero metadata from file if provided, otherwise use default mappings
        if heroes_file and os.path.exists(heroes_file):
            self.load_hero_metadata(heroes_file)
        else:
            self.initialize_default_metadata()

    def initialize_default_metadata(self):
        # Add basic hero attribute information (partial example)
        # Full list should be added from Dota 2 wiki or official sources
        attributes = {'str': 1, 'agi': 2, 'int': 3}

        # Example mappings (add all heroes in actual implementation)
        hero_data = {
            # Format: hero_id: [primary_attribute, [roles]]
            1: ['str', ['carry', 'durable']],  # Anti-Mage
            2: ['agi', ['carry', 'escape']],  # Axe
            3: ['int', ['support', 'disabler']],  # Bane
            # Add more heroes...
        }

        for hero_id, data in hero_data.items():
            self.hero_attributes[hero_id] = attributes.get(data[0], 0)
            self.hero_roles[hero_id] = data[1]

    def load_hero_metadata(self, file_path):
        # Load hero data from external file
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                hero_id = row['hero_id']
                self.hero_attributes[hero_id] = row.get('primary_attribute', 0)
                self.hero_roles[hero_id] = row.get('roles', '').split(',')
        except Exception as e:
            print(f"Error loading hero metadata: {e}")
            self.initialize_default_metadata()

    def update_pick_rates(self, pick_rates_dict, win_rates_dict):
        self.hero_pick_rates = pick_rates_dict
        self.hero_win_rates = win_rates_dict

    def get_hero_feature_dict(self, hero_id):
        return {
            'attribute': self.hero_attributes.get(hero_id, 0),
            'roles': self.hero_roles.get(hero_id, []),
            'pick_rate': self.hero_pick_rates.get(hero_id, 0),
            'win_rate': self.hero_win_rates.get(hero_id, 0)
        }


class DotaDatasetCollector:
    def __init__(self, output_dir='dota2_promatches_dataset'):
        self.output_dir = output_dir
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/batches", exist_ok=True)
        os.makedirs(f"{output_dir}/daily", exist_ok=True)

        # API endpoints
        self.PRO_MATCHES_URL = "https://api.opendota.com/api/proMatches"
        self.PUBLIC_MATCHES_URL = "https://api.opendota.com/api/publicMatches"
        self.PARSED_MATCHES_URL = "https://api.opendota.com/api/parsedMatches"
        self.MATCH_DETAILS_URL = "https://api.opendota.com/api/matches/{}"
        self.HEROES_URL = "https://api.opendota.com/api/heroes"

        # Flag to determine which API to use
        self.use_pro_matches = True

        # Track collected matches
        self.collected_match_ids = set()
        self.batch_size = 1000
        self.current_batch = []
        self.total_collected = 0
        self.daily_collected = 0
        self.batch_counter = 0

        # Pagination tracking
        self.last_match_id = None
        self.pagination_state_file = f"{output_dir}/pagination_state.pkl"

        # Rate limiting controls
        self.base_delay = 1.0
        self.current_delay = self.base_delay
        self.max_delay = 30.0
        self.consecutive_429s = 0
        self.last_request_time = datetime.now() - timedelta(seconds=10)  # Ensure we can make a request immediately

        # Daily tracking
        self.current_date = datetime.now().date()
        self.daily_request_count = 0
        self.daily_limit = 25000  # Reasonable limit considering 2 requests per match

        # Hero metadata
        self.hero_metadata_file = f"{output_dir}/hero_metadata.csv"

        # Load existing match IDs from all historical batches
        self.load_all_existing_match_ids()
        self.load_pagination_state()

        # Ensure we have hero metadata
        if not os.path.exists(self.hero_metadata_file):
            self.download_hero_metadata()

    def load_all_existing_match_ids(self):
        """Load all match IDs from historical batch files and daily files"""
        # Load from batch files
        batch_files = []
        if os.path.exists(f"{self.output_dir}/batches"):
            batch_files = [f for f in os.listdir(f"{self.output_dir}/batches") if
                           f.startswith("matches_batch_") and f.endswith(".csv")]

        # Load from daily files
        daily_files = []
        if os.path.exists(f"{self.output_dir}/daily"):
            daily_files = [f for f in os.listdir(f"{self.output_dir}/daily") if
                           f.startswith("matches_") and f.endswith(".csv")]

        all_files = batch_files + daily_files

        if all_files:
            for file in all_files:
                directory = "batches" if file.startswith("matches_batch_") else "daily"
                try:
                    file_path = f"{self.output_dir}/{directory}/{file}"
                    df = pd.read_csv(file_path)
                    if 'match_id' in df.columns:
                        self.collected_match_ids.update(df['match_id'].astype(str).tolist())

                    if file.startswith("matches_batch_"):
                        batch_num = int(file.split('_')[2].split('.')[0])
                        self.batch_counter = max(self.batch_counter, batch_num + 1)
                except Exception as e:
                    print(f"Error loading file {file}: {str(e)}")

            self.total_collected = len(self.collected_match_ids)
            print(f"Loaded {self.total_collected} existing match IDs from {len(all_files)} files")
        else:
            print("No existing match files found. Starting fresh collection.")

    def load_pagination_state(self):
        """Load pagination state from file"""
        if os.path.exists(self.pagination_state_file):
            try:
                with open(self.pagination_state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.last_match_id = state.get('last_match_id')
                    self.daily_request_count = state.get('daily_request_count', 0)

                    # Reset if we're on a new day
                    saved_date = state.get('date')
                    if saved_date != self.current_date:
                        print(f"New day detected. Resetting daily counters.")
                        self.daily_request_count = 0
                        self.daily_collected = 0

                print(
                    f"Loaded pagination state: last_match_id={self.last_match_id}, daily_request_count={self.daily_request_count}")
            except Exception as e:
                print(f"Error loading pagination state: {str(e)}")
                self.last_match_id = None
                self.daily_request_count = 0

    def save_pagination_state(self):
        """Save pagination state to file"""
        state = {
            'last_match_id': self.last_match_id,
            'daily_request_count': self.daily_request_count,
            'date': self.current_date,
            'total_collected': self.total_collected,
            'daily_collected': self.daily_collected
        }
        try:
            with open(self.pagination_state_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"Error saving pagination state: {str(e)}")

    def download_hero_metadata(self):
        """Download hero metadata from OpenDota API"""
        print("Downloading hero metadata...")

        try:
            # Respect rate limiting
            self.wait_for_rate_limit()

            response = requests.get(self.HEROES_URL)
            self.daily_request_count += 1
            self.last_request_time = datetime.now()

            if response.status_code == 200:
                heroes_data = response.json()

                # Process hero data
                hero_records = []
                for hero in heroes_data:
                    hero_record = {
                        'hero_id': hero['id'],
                        'name': hero['localized_name'],
                        'primary_attribute': hero['primary_attr'],
                        'attack_type': hero['attack_type'],
                        'roles': ','.join(hero['roles']),
                    }
                    hero_records.append(hero_record)

                # Save to CSV
                heroes_df = pd.DataFrame(hero_records)
                heroes_df.to_csv(self.hero_metadata_file, index=False)
                print(f"Saved metadata for {len(heroes_df)} heroes to {self.hero_metadata_file}")
                return self.hero_metadata_file
            else:
                print(f"Failed to download hero metadata: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading hero metadata: {str(e)}")
            return None

    def collect_matches(self, daily_target=50000, max_daily_requests=25000, use_pro_matches=True):
        """Collect Dota 2 matches with daily target and pagination"""
        # Set which API to use
        self.use_pro_matches = use_pro_matches
        api_type = "pro" if self.use_pro_matches else "public"

        # Check for a new day
        current_date = datetime.now().date()
        if current_date != self.current_date:
            print(f"New day detected. Resetting daily counters.")
            self.current_date = current_date
            self.daily_request_count = 0
            self.daily_collected = 0

        print(f"Starting collection of {daily_target} Dota 2 {api_type} matches for today ({self.current_date})")
        print(f"All-time collected matches: {self.total_collected}")
        print(f"Today's collected matches so far: {self.daily_collected}")

        # Set API request limit
        self.daily_limit = max_daily_requests

        # Debug: Print API URL being used
        api_url = self.PRO_MATCHES_URL if self.use_pro_matches else self.PUBLIC_MATCHES_URL
        print(f"Using API endpoint: {api_url}")

        # Check if we've already hit today's target
        if self.daily_collected >= daily_target:
            print(f"Daily target of {daily_target} matches already reached!")
            self.create_ml_dataset()
            self.create_hero_frequency_analysis()
            return

        print(f"Need to collect {daily_target - self.daily_collected} more matches today")
        print(f"Daily request limit: {self.daily_limit}, Used today: {self.daily_request_count}")

        if self.last_match_id:
            print(f"Continuing from last match ID: {self.last_match_id}")

        start_time = datetime.now()
        request_counter = 0

        try:
            while self.daily_collected < daily_target:
                # Check if we've hit the daily API request limit
                if self.daily_request_count >= self.daily_limit:
                    print(f"\nDaily API request limit reached ({self.daily_limit}). Stopping until tomorrow.")
                    print(f"Today's progress: {self.daily_collected}/{daily_target} matches collected.")

                    # Save current state before exiting
                    if self.current_batch:
                        self.save_current_batch()
                    self.save_pagination_state()
                    self.create_daily_file()

                    # Calculate next possible run time
                    minutes_to_wait = 15  # Wait 15 minutes and try again
                    next_run_time = datetime.now() + timedelta(minutes=minutes_to_wait)

                    print(
                        f"Try again in approximately {minutes_to_wait} minutes (at {next_run_time.strftime('%H:%M:%S')})")
                    return

                try:
                    # Display stats periodically
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if request_counter % 5 == 0 and elapsed > 0:
                        rate = self.daily_collected / elapsed if elapsed > 0 else 0
                        remaining_count = daily_target - self.daily_collected
                        estimated_seconds = remaining_count / rate if rate > 0 else 0
                        estimated_time = time.strftime('%H:%M:%S', time.gmtime(estimated_seconds))
                        print(
                            f"\nFetch #{request_counter + 1} - Today's progress: {self.daily_collected}/{daily_target}")
                        print(f"API Requests today: {self.daily_request_count}/{self.daily_limit}")
                        print(f"Rate: {rate:.2f} matches/sec, Est. time remaining: {estimated_time}")
                        print(f"Current delay between requests: {self.current_delay:.2f}s")
                    else:
                        print(
                            f"\rFetch #{request_counter + 1} - Today: {self.daily_collected}/{daily_target} - Requests: {self.daily_request_count}/{self.daily_limit}",
                            end="")

                    # Respect rate limiting delay
                    self.wait_for_rate_limit()

                    # Fetch pro matches with pagination
                    matches, status_code = self.fetch_pro_matches()
                    request_counter += 1
                    self.daily_request_count += 1

                    # Save state periodically
                    if request_counter % 10 == 0:
                        self.save_pagination_state()

                    # Adjust delay based on response
                    self.adjust_delay(status_code)

                    if not matches:
                        # If no matches are returned, we may have reached the end
                        # Reset last_match_id to get the most recent matches again
                        if status_code == 200:
                            print("\nNo more matches available. Resetting pagination to get recent matches.")
                            self.last_match_id = None
                            time.sleep(5)  # Wait a bit before retrying
                        continue

                    # Process matches
                    new_matches = self.process_matches(matches, daily_target)

                    # Update pagination for next request
                    if matches and len(matches) > 0 and 'match_id' in matches[-1]:
                        self.last_match_id = int(matches[-1]['match_id'])

                    # Report new matches
                    if new_matches > 0:
                        print(f", Added {new_matches} new matches", end="")

                        # Save batch if it reaches batch size
                        if len(self.current_batch) >= self.batch_size:
                            self.save_current_batch()
                    elif len(matches) > 0:
                        print(f", All {len(matches)} matches already collected", end="")

                    # Check if we've reached our daily target
                    if self.daily_collected >= daily_target:
                        break

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"\nError in collection loop: {str(e)}")
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\n\nCollection interrupted by user. Saving current progress...")

        # Save any remaining matches
        if self.current_batch:
            self.save_current_batch()

        # Save pagination state
        self.save_pagination_state()

        # Create the daily file
        self.create_daily_file()

        # Create ML dataset and hero analysis
        self.create_ml_dataset()
        self.create_hero_frequency_analysis()

        # Display total time
        total_time = datetime.now() - start_time
        print(f"\nCollection completed in {total_time}. Today's total matches: {self.daily_collected}")
        print(f"All-time total matches: {self.total_collected}")

    def wait_for_rate_limit(self):
        """Wait for the appropriate amount of time between requests"""
        elapsed = (datetime.now() - self.last_request_time).total_seconds()
        if elapsed < self.current_delay:
            wait_time = self.current_delay - elapsed
            time.sleep(wait_time)

    def adjust_delay(self, status_code):
        """Adjust the delay between requests based on API response"""
        if status_code == 429:  # Too Many Requests
            self.consecutive_429s += 1
            # Exponential backoff
            self.current_delay = min(self.current_delay * 2, self.max_delay)
            print(f"\nHit rate limit (429 error). Increasing delay to {self.current_delay:.2f}s")

            # If we get too many 429s in a row, take a longer break
            if self.consecutive_429s >= 3:
                cooldown = 60
                print(f"Too many rate limit errors. Taking a {cooldown}s break...")
                time.sleep(cooldown)
                self.consecutive_429s = 0
        elif status_code == 200:
            # Reset counter on success
            self.consecutive_429s = 0
            # Gradually decrease delay but don't go below base
            if self.current_delay > self.base_delay:
                self.current_delay = max(self.current_delay * 0.95, self.base_delay)

    def fetch_pro_matches(self):
        """Fetch matches from the API with pagination"""
        self.last_request_time = datetime.now()

        # Select appropriate API based on flag
        api_url = self.PRO_MATCHES_URL if self.use_pro_matches else self.PUBLIC_MATCHES_URL

        params = {"limit": 100}  # Max allowed by API

        # Add pagination if we have a last match ID
        if self.last_match_id:
            params["less_than_match_id"] = self.last_match_id

        try:
            response = requests.get(api_url, params=params)
            status_code = response.status_code

            if status_code == 200:
                data = response.json()
                # Print first match ID if available for debugging
                if data and len(data) > 0:
                    print(f"\nReceived {len(data)} matches. First match ID: {data[0].get('match_id')}")
                else:
                    print("\nNo matches returned from API")

                    # Try alternate API if no matches found
                    if len(data) == 0 and self.use_pro_matches:
                        print("Trying parsed matches API as fallback...")
                        self.wait_for_rate_limit()  # Respect rate limiting
                        fallback_response = requests.get(self.PARSED_MATCHES_URL, params=params)
                        self.daily_request_count += 1
                        self.last_request_time = datetime.now()

                        if fallback_response.status_code == 200:
                            fallback_data = fallback_response.json()
                            if fallback_data and len(fallback_data) > 0:
                                print(f"Found {len(fallback_data)} matches from fallback API")
                                return fallback_data, fallback_response.status_code

                return data, status_code
            else:
                print(f"\nAPI Error {status_code}: Failed to fetch matches")
                if hasattr(response, 'text'):
                    print(f"Response: {response.text[:200]}...")
                return [], status_code
        except Exception as e:
            print(f"\nError fetching matches: {str(e)}")
            return [], 0

    def process_matches(self, matches, daily_target):
        """Process a batch of matches, filter out already collected ones"""
        new_matches = 0
        skipped_format = 0
        skipped_duplicate = 0

        for match in matches:
            if not match or 'match_id' not in match:
                print("Skipping: Missing match_id")
                continue

            match_id = str(match['match_id'])

            # Skip if already collected
            if match_id in self.collected_match_ids:
                skipped_duplicate += 1
                continue

            # For pro matches, we need to fetch the detailed match data
            try:
                # Wait for rate limit before making the second request
                self.wait_for_rate_limit()

                match_url = self.MATCH_DETAILS_URL.format(match_id)
                response = requests.get(match_url)
                self.daily_request_count += 1
                self.last_request_time = datetime.now()

                if response.status_code != 200:
                    print(f"Failed to get match details for {match_id}: HTTP {response.status_code}")
                    continue

                match_details = response.json()

                # Extract heroes from players data
                if 'players' not in match_details or not isinstance(match_details['players'], list) or len(
                        match_details['players']) != 10:
                    print(
                        f"Match {match_id} doesn't have valid player data: {len(match_details.get('players', []))} players")
                    skipped_format += 1
                    continue

                # Check if radiant_win exists
                if 'radiant_win' not in match_details:
                    print(f"Match {match_id} missing radiant_win field")
                    skipped_format += 1
                    continue

                radiant_heroes = []
                dire_heroes = []

                for player in match_details['players']:
                    if 'hero_id' not in player:
                        continue

                    is_radiant = player.get('isRadiant', player.get('player_slot', 0) < 128)

                    if is_radiant:
                        radiant_heroes.append(player['hero_id'])
                    else:
                        dire_heroes.append(player['hero_id'])

                # Skip if we don't have 5 heroes per team
                if len(radiant_heroes) != 5 or len(dire_heroes) != 5:
                    print(
                        f"Match {match_id} doesn't have 5 heroes per team: R={len(radiant_heroes)}, D={len(dire_heroes)}")
                    skipped_format += 1
                    continue

                # Create processed match record
                processed_match = {
                    'match_id': match_id,
                    'start_time': match_details.get('start_time', match.get('start_time')),
                    'duration': match_details.get('duration', match.get('duration')),
                    'radiant_win': match_details['radiant_win'],
                    'radiant_score': match_details.get('radiant_score', match.get('radiant_score')),
                    'dire_score': match_details.get('dire_score', match.get('dire_score')),
                    'league_name': match.get('league_name', ''),
                    'radiant_heroes': radiant_heroes,
                    'dire_heroes': dire_heroes
                }

                # Add to current batch
                self.current_batch.append(processed_match)
                self.collected_match_ids.add(match_id)
                self.total_collected += 1
                self.daily_collected += 1
                new_matches += 1

                # Break early if we've reached our daily target
                if self.daily_collected >= daily_target:
                    break

            except Exception as e:
                print(f"Error processing match {match_id}: {str(e)}")

        # Report skipped matches
        if skipped_duplicate > 0:
            print(f", Skipped {skipped_duplicate} already collected matches", end="")
        if skipped_format > 0:
            print(f", Skipped {skipped_format} matches due to format issues", end="")

        return new_matches

    def save_current_batch(self):
        """Save the current batch of matches to a CSV file"""
        if not self.current_batch:
            return

        batch_df = pd.DataFrame(self.current_batch)

        # Convert hero lists to strings for CSV storage
        batch_df['radiant_heroes'] = batch_df['radiant_heroes'].apply(
            lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
        batch_df['dire_heroes'] = batch_df['dire_heroes'].apply(
            lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)

        # Save to CSV
        batch_file = f"{self.output_dir}/batches/matches_batch_{self.batch_counter}.csv"
        batch_df.to_csv(batch_file, index=False)

        print(f"\nSaved batch #{self.batch_counter} with {len(self.current_batch)} matches to {batch_file}")

        # Clear current batch and increment counter
        self.current_batch = []
        self.batch_counter += 1

        # Save pagination state
        self.save_pagination_state()

    def create_daily_file(self):
        """Create a daily summary file with today's matches"""
        today_str = self.current_date.strftime('%Y%m%d')
        daily_file = f"{self.output_dir}/daily/matches_{today_str}.csv"

        # Skip if no matches collected today
        if self.daily_collected == 0:
            return

        print(f"\nCreating daily summary file for {self.current_date}...")

        # Get all batches for processing
        batch_files = []
        if os.path.exists(f"{self.output_dir}/batches"):
            batch_files = [f for f in os.listdir(f"{self.output_dir}/batches")
                           if f.startswith("matches_batch_") and f.endswith(".csv")]

        if not batch_files:
            print("No batch files found")
            return

        # Load all matches and filter for today
        today_matches = []
        start_of_day = int(datetime.combine(self.current_date, datetime.min.time()).timestamp())
        end_of_day = int(datetime.combine(self.current_date, datetime.max.time()).timestamp())

        for batch_file in batch_files:
            try:
                file_path = f"{self.output_dir}/batches/{batch_file}"
                df = pd.read_csv(file_path)

                # Filter matches from today based on start_time
                if 'start_time' in df.columns:
                    today_df = df[(df['start_time'] >= start_of_day) & (df['start_time'] <= end_of_day)]
                    if not today_df.empty:
                        today_matches.append(today_df)
            except Exception as e:
                print(f"Error processing batch file {batch_file}: {str(e)}")

        # Combine and save
        if today_matches:
            combined_df = pd.concat(today_matches, ignore_index=True)
            combined_df.to_csv(daily_file, index=False)
            print(f"Saved {len(combined_df)} matches from today to {daily_file}")
        else:
            print("No matches from today found in batches")

    def create_ml_dataset(self):
        """Create an ML-ready dataset with enhanced features"""
        # First create hero statistics
        self.create_hero_frequency_analysis()

        # Create hero metadata object
        hero_metadata = DotaHeroMetadata(self.hero_metadata_file)

        # Load pick rates and win rates
        hero_stats_path = f"{self.output_dir}/hero_statistics.csv"
        if os.path.exists(hero_stats_path):
            hero_stats = pd.read_csv(hero_stats_path)
            pick_rates = dict(zip(hero_stats['hero_id'], hero_stats['pick_rate']))
            win_rates = dict(zip(hero_stats['hero_id'], hero_stats['win_rate']))
            hero_metadata.update_pick_rates(pick_rates, win_rates)

        # Load all batch files
        batch_files = []
        if os.path.exists(f"{self.output_dir}/batches"):
            batch_files = [f for f in os.listdir(f"{self.output_dir}/batches")
                           if f.startswith("matches_batch_") and f.endswith(".csv")]

        if not batch_files:
            print("No batch files found")
            return

        ml_records = []
        ml_records_with_draft = []

        print("\nCreating ML-ready dataset...")

        for batch_file in batch_files:
            try:
                file_path = f"{self.output_dir}/batches/{batch_file}"
                df = pd.read_csv(file_path)

                # Process each match
                for _, row in df.iterrows():
                    # Convert hero strings back to lists
                    radiant_heroes = row['radiant_heroes']
                    dire_heroes = row['dire_heroes']

                    if isinstance(radiant_heroes, str):
                        radiant_heroes = [int(h) for h in radiant_heroes.split(',') if h]

                    if isinstance(dire_heroes, str):
                        dire_heroes = [int(h) for h in dire_heroes.split(',') if h]

                    # Skip if we don't have 5 heroes per team
                    if len(radiant_heroes) != 5 or len(dire_heroes) != 5:
                        continue

                    # Create records with draft order simulation (2-2-1 pattern)
                    all_heroes = []

                    # First phase - both teams pick 2 heroes
                    all_heroes.append(('radiant', radiant_heroes[0]))
                    all_heroes.append(('radiant', radiant_heroes[1]))
                    all_heroes.append(('dire', dire_heroes[0]))
                    all_heroes.append(('dire', dire_heroes[1]))

                    # Second phase - both teams pick 2 more heroes
                    all_heroes.append(('radiant', radiant_heroes[2]))
                    all_heroes.append(('radiant', radiant_heroes[3]))
                    all_heroes.append(('dire', dire_heroes[2]))
                    all_heroes.append(('dire', dire_heroes[3]))

                    # Third phase - both teams pick final hero
                    all_heroes.append(('radiant', radiant_heroes[4]))
                    all_heroes.append(('dire', dire_heroes[4]))

                    # Create enhanced record with all features
                    enhanced_record = {
                        'match_id': row['match_id'],
                        'radiant_win': row['radiant_win'],
                        'duration': row.get('duration', 0),
                        'league_name': row.get('league_name', '')
                    }

                    # Add heroes with draft order and features
                    for pick_num, (team, hero_id) in enumerate(all_heroes, 1):
                        # Add basic hero ID
                        enhanced_record[f'pick_{pick_num}_team'] = team
                        enhanced_record[f'pick_{pick_num}_hero'] = hero_id

                        # Add hero features
                        hero_features = hero_metadata.get_hero_feature_dict(hero_id)
                        enhanced_record[f'pick_{pick_num}_attribute'] = hero_features['attribute']
                        enhanced_record[f'pick_{pick_num}_pick_rate'] = hero_features['pick_rate']
                        enhanced_record[f'pick_{pick_num}_win_rate'] = hero_features['win_rate']

                        # Store roles as comma-separated string
                        role_str = ','.join(hero_features['roles']) if hero_features['roles'] else ''
                        enhanced_record[f'pick_{pick_num}_roles'] = role_str

                    ml_records_with_draft.append(enhanced_record)

                    # Create traditional record format
                    ml_record = {
                        'match_id': row['match_id'],
                        'radiant_win': row['radiant_win'],
                        'radiant_hero_1': radiant_heroes[0],
                        'radiant_hero_2': radiant_heroes[1],
                        'radiant_hero_3': radiant_heroes[2],
                        'radiant_hero_4': radiant_heroes[3],
                        'radiant_hero_5': radiant_heroes[4],
                        'dire_hero_1': dire_heroes[0],
                        'dire_hero_2': dire_heroes[1],
                        'dire_hero_3': dire_heroes[2],
                        'dire_hero_4': dire_heroes[3],
                        'dire_hero_5': dire_heroes[4],
                    }

                    # Add pick rates and win rates
                    for i, hero_id in enumerate(radiant_heroes, 1):
                        ml_record[f'radiant_hero_{i}_pick_rate'] = hero_metadata.hero_pick_rates.get(hero_id, 0)
                        ml_record[f'radiant_hero_{i}_win_rate'] = hero_metadata.hero_win_rates.get(hero_id, 0)
                        ml_record[f'radiant_hero_{i}_attribute'] = hero_metadata.hero_attributes.get(hero_id, 0)

                    for i, hero_id in enumerate(dire_heroes, 1):
                        ml_record[f'dire_hero_{i}_pick_rate'] = hero_metadata.hero_pick_rates.get(hero_id, 0)
                        ml_record[f'dire_hero_{i}_win_rate'] = hero_metadata.hero_win_rates.get(hero_id, 0)
                        ml_record[f'dire_hero_{i}_attribute'] = hero_metadata.hero_attributes.get(hero_id, 0)

                    ml_records.append(ml_record)
            except Exception as e:
                print(f"Error processing batch file {batch_file}: {str(e)}")

        # Save ML-ready datasets
        if ml_records:
            ml_df = pd.DataFrame(ml_records)
            ml_df.to_csv(f"{self.output_dir}/ml_ready_data.csv", index=False)
            print(f"Saved traditional ML dataset with {len(ml_records)} matches")

        if ml_records_with_draft:
            ml_draft_df = pd.DataFrame(ml_records_with_draft)
            ml_draft_df.to_csv(f"{self.output_dir}/ml_ready_data_with_draft.csv", index=False)
            print(f"Saved enhanced ML dataset with draft order for {len(ml_records_with_draft)} matches")

    def create_hero_frequency_analysis(self):
        """Create an analysis of hero pick frequencies and win rates"""
        # Check if we have a heroes.csv file with all hero appearances
        hero_file = f"{self.output_dir}/heroes.csv"
        if os.path.exists(hero_file):
            print("\nCreating hero statistics from heroes.csv...")
            heroes_df = pd.read_csv(hero_file)

            # Group by hero_id to get pick and win rates
            hero_stats = heroes_df.groupby('hero_id').agg(
                pick_count=('match_id', 'count'),
                win_count=('won', 'sum')
            ).reset_index()

            # Calculate win rates
            hero_stats['win_rate'] = hero_stats['win_count'] / hero_stats['pick_count']

            # Calculate pick rates
            total_matches = len(heroes_df['match_id'].unique())
            hero_stats['pick_rate'] = hero_stats['pick_count'] / (total_matches * 10)  # 10 heroes per match

            # Sort by pick count
            hero_stats = hero_stats.sort_values('pick_count', ascending=False)

            # Save hero stats
            hero_stats.to_csv(f"{self.output_dir}/hero_statistics.csv", index=False)
            print(f"Saved hero statistics for {len(hero_stats)} heroes")
            return

        # If no heroes.csv exists, extract from batch files
        print("\nCreating hero statistics from match data...")

        # Load all batch files
        batch_files = []
        if os.path.exists(f"{self.output_dir}/batches"):
            batch_files = [f for f in os.listdir(f"{self.output_dir}/batches")
                           if f.startswith("matches_batch_") and f.endswith(".csv")]

        if not batch_files:
            print("No batch files found")
            return

        # Count hero frequencies
        hero_counts = {}
        win_counts = {}
        hero_records = []

        for batch_file in batch_files:
            try:
                file_path = f"{self.output_dir}/batches/{batch_file}"
                df = pd.read_csv(file_path)

                # Process each match
                for _, row in df.iterrows():
                    # Convert hero strings back to lists
                    radiant_heroes = row['radiant_heroes']
                    dire_heroes = row['dire_heroes']

                    if isinstance(radiant_heroes, str):
                        radiant_heroes = [int(h) for h in radiant_heroes.split(',') if h]

                    if isinstance(dire_heroes, str):
                        dire_heroes = [int(h) for h in dire_heroes.split(',') if h]

                    # Skip if we don't have proper data
                    if len(radiant_heroes) != 5 or len(dire_heroes) != 5:
                        continue

                    # Process radiant heroes
                    for hero_id in radiant_heroes:
                        if hero_id not in hero_counts:
                            hero_counts[hero_id] = 0
                            win_counts[hero_id] = 0

                        hero_counts[hero_id] += 1

                        # Add hero record
                        hero_records.append({
                            'match_id': row['match_id'],
                            'hero_id': hero_id,
                            'team': 'radiant',
                            'won': row['radiant_win']
                        })

                        # Count wins
                        if row['radiant_win']:
                            win_counts[hero_id] += 1

                    # Process dire heroes
                    for hero_id in dire_heroes:
                        if hero_id not in hero_counts:
                            hero_counts[hero_id] = 0
                            win_counts[hero_id] = 0

                        hero_counts[hero_id] += 1

                        # Add hero record
                        hero_records.append({
                            'match_id': row['match_id'],
                            'hero_id': hero_id,
                            'team': 'dire',
                            'won': not row['radiant_win']
                        })

                        # Count wins (dire wins are !radiant_win)
                        if not row['radiant_win']:
                            win_counts[hero_id] += 1

            except Exception as e:
                print(f"Error processing batch file {batch_file}: {str(e)}")

        # Save hero records to CSV for future use
        if hero_records:
            heroes_df = pd.DataFrame(hero_records)
            heroes_df.to_csv(f"{self.output_dir}/heroes.csv", index=False)
            print(f"Saved {len(heroes_df)} hero records")

        # Calculate win rates
        win_rates = {}
        for hero_id in hero_counts:
            win_rates[hero_id] = win_counts[hero_id] / hero_counts[hero_id] if hero_counts[hero_id] > 0 else 0

        # Create hero stats dataframe
        total_matches = len(set([record['match_id'] for record in hero_records])) if hero_records else 0

        hero_stats = []
        for hero_id in hero_counts:
            hero_stats.append({
                'hero_id': hero_id,
                'pick_count': hero_counts[hero_id],
                'win_count': win_counts[hero_id],
                'win_rate': win_rates[hero_id],
                'pick_rate': hero_counts[hero_id] / (total_matches * 10) if total_matches > 0 else 0
            })

        # Save hero stats
        if hero_stats:
            hero_stats_df = pd.DataFrame(hero_stats)
            hero_stats_df = hero_stats_df.sort_values('pick_count', ascending=False)
            hero_stats_df.to_csv(f"{self.output_dir}/hero_statistics.csv", index=False)
            print(f"Saved hero statistics for {len(hero_stats_df)} heroes")


# Example usage
if __name__ == "__main__":
    collector = DotaDatasetCollector(output_dir="dota2_promatches")

    # Define collection parameters
    collector.collect_matches(
        daily_target=50000,  # Target 50,000 unique matches daily
        max_daily_requests=25000,  # Allow up to 25,000 API requests (we need 2 per match)
        use_pro_matches=True  # Use pro matches API
    )