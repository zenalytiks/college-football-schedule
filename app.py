import pandas as pd
import numpy as np
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime
from dash_calendar_timeline import DashCalendarTimeline
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock
import pickle
import os
from functools import wraps
import redis
import json
from typing import Optional, Dict, Any, Tuple, List

from banner import generate_custom_scoreboard
import base64
# from cfbd_data_reader import read_data
from datetime import datetime

# Get current date
CURRENT_DATE = datetime.now()

# Server-side cache configuration
CACHE_TYPE = "redis"  # Options: "file", "redis", "memory"
CACHE_DIR = "./cache"
CACHE_TTL = 3600  # 1 hour in seconds
REDIS_URL = "redis://localhost:6379/0"  # Configure as needed


class ServerCache:
    """Server-side cache implementation supporting multiple backends."""
    
    def __init__(self, cache_type: str = "redis", cache_dir: str = "./cache", 
                 ttl: int = 3600, redis_url: str = None):
        self.cache_type = cache_type
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.redis_client = None
        self.memory_cache = {}
        self.memory_cache_timestamps = {}
        self.cache_lock = Lock()
        
        if cache_type == "file" and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        elif cache_type == "redis":
            try:
                import redis
                self.redis_client = redis.from_url(redis_url or REDIS_URL)
                self.redis_client.ping()  # Test connection
            except Exception as e:
                print(f"Redis connection failed, falling back to file cache: {e}")
                self.cache_type = "file"
    
    def _get_cache_key(self, key: str) -> str:
        """Generate a standardized cache key."""
        return f"cfb_app_{hashlib.md5(key.encode()).hexdigest()}"
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._get_cache_key(key)
        
        try:
            if self.cache_type == "redis" and self.redis_client:
                data = self.redis_client.get(cache_key)
                if data:
                    cached_data = pickle.loads(data)
                    if not self._is_expired(cached_data.get('timestamp', 0)):
                        return cached_data.get('value')
                    else:
                        self.redis_client.delete(cache_key)
                        
            elif self.cache_type == "file":
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    if not self._is_expired(cached_data.get('timestamp', 0)):
                        return cached_data.get('value')
                    else:
                        os.remove(cache_file)
                        
            elif self.cache_type == "memory":
                with self.cache_lock:
                    if cache_key in self.memory_cache:
                        timestamp = self.memory_cache_timestamps.get(cache_key, 0)
                        if not self._is_expired(timestamp):
                            return self.memory_cache[cache_key]
                        else:
                            del self.memory_cache[cache_key]
                            del self.memory_cache_timestamps[cache_key]
                            
        except Exception as e:
            print(f"Cache get error for key {key}: {e}")
            
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        cache_key = self._get_cache_key(key)
        cached_data = {
            'value': value,
            'timestamp': time.time()
        }
        
        try:
            if self.cache_type == "redis" and self.redis_client:
                self.redis_client.setex(
                    cache_key, 
                    self.ttl, 
                    pickle.dumps(cached_data)
                )
                
            elif self.cache_type == "file":
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
                    
            elif self.cache_type == "memory":
                with self.cache_lock:
                    self.memory_cache[cache_key] = value
                    self.memory_cache_timestamps[cache_key] = time.time()
                    
        except Exception as e:
            print(f"Cache set error for key {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            if self.cache_type == "redis" and self.redis_client is not None:
                keys = self.redis_client.keys("cfb_app_*")
                if keys:
                    self.redis_client.delete(*keys)
                print(f"✓ Redis cache cleared: {len(keys)} keys removed")
                    
            elif self.cache_type == "file":
                if os.path.exists(self.cache_dir):
                    files_removed = 0
                    for filename in os.listdir(self.cache_dir):
                        if filename.startswith("cfb_app_") and filename.endswith(".pkl"):
                            os.remove(os.path.join(self.cache_dir, filename))
                            files_removed += 1
                    print(f"✓ File cache cleared: {files_removed} files removed")
                        
            elif self.cache_type == "memory":
                with self.cache_lock:
                    cache_count = len(self.memory_cache)
                    self.memory_cache.clear()
                    self.memory_cache_timestamps.clear()
                    print(f"✓ Memory cache cleared: {cache_count} entries removed")
                    
        except Exception as e:
            print(f"❌ Cache clear error: {e}")
            import traceback
            traceback.print_exc()


def cache_result(cache_instance: ServerCache, key_func=None):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)
            return result
        return wrapper
    return decorator


class CFBGuideApp:
    def __init__(self, max_workers=4, cache_type="redis", cache_ttl=3600):
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=0.5"}]
        )
        self.server = self.app.server
        
        # Initialize server-side cache
        self.server_cache = ServerCache(
            cache_type=cache_type,
            cache_dir=CACHE_DIR,
            ttl=cache_ttl,
            redis_url=REDIS_URL
        )
        
        self.df = self._load_and_prepare_data()
        self.min_date, self.max_date = self._get_date_bounds()
        
        # Threading configuration
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Thread-safe cache for SVG scoreboards (in-memory for performance)
        self._scoreboard_cache = {}
        self._cache_lock = Lock()
        
        # Pre-computation cache for game data processing (in-memory for performance)
        self._game_data_cache = {}
        self._game_data_lock = Lock()
        
        self._setup_layout()
        self._setup_callbacks()

    def _load_and_prepare_data(self):
        """Load and prepare the data with optimized column operations."""
        # Check if processed data is cached
        data_cache_key = "processed_dataframe"
        cached_df = self.server_cache.get(data_cache_key)
        
        if cached_df is not None:
            print("Loading processed data from cache")
            return cached_df
        
        print("Processing raw data...")
        # df = read_data() # Read data from API
        df = pd.read_csv('./data/cfbd_data.csv') # Read data from local file
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = df['start_date'] + pd.Timedelta(hours=3, minutes=30)
        
        # Pre-compute timestamp conversions
        df['start_time_ms'] = df['start_date'].astype("int64") // 10**6
        df['end_time_ms'] = df['end_date'].astype("int64") // 10**6
        
        # Cache the processed dataframe
        self.server_cache.set(data_cache_key, df)
        print("Processed data cached")
        
        return df

    def _get_date_bounds(self):
        """Get min and max dates from the dataset."""
        bounds_cache_key = "date_bounds"
        cached_bounds = self.server_cache.get(bounds_cache_key)
        
        if cached_bounds is not None:
            return cached_bounds
        
        date_col = self.df['start_date']
        bounds = (date_col.min(), date_col.max())
        self.server_cache.set(bounds_cache_key, bounds)
        return bounds

    def _setup_layout(self):
        """Setup the application layout with college football dark theme."""
        self.app.layout = dbc.Container([
            # Navigation with college football theme
            dbc.NavbarSimple(
                brand="🏈 College Football Schedule",
                brand_href="#",
                color="success",  # Forest green for college football
                dark=True,
                className="shadow-lg border-bottom border-warning",
                style={
                    'box-shadow': '0 4px 15px rgba(0,0,0,0.3)'
                },
                fluid=True
            ),
            
            # Cache status indicator
            dbc.Alert([
                html.I(className="fas fa-database me-2"),
                html.Span("Server-side caching enabled", className="fw-bold"),
                html.Span(f" ({self.server_cache.cache_type.upper()})", className="text-muted ms-2")
            ], color="info", className="mt-2 mb-2", dismissable=True),
            
            # Main content container with dark theme
            dbc.Container([
                # Date picker with enhanced styling
                dbc.Stack([
                            html.Label(
                                "Select Game Date", 
                                className="text-light fw-bold mb-2",
                                style={'color': '#f8f9fa', 'font-size': '1.1rem','padding-right': '50px'}
                            ),
                            dcc.DatePickerSingle(
                                id='date-picker',
                                min_date_allowed=datetime(self.min_date.year, self.min_date.month, self.min_date.day),
                                max_date_allowed=datetime(self.max_date.year, self.max_date.month, self.max_date.day),
                                initial_visible_month=datetime(self.min_date.year, self.min_date.month, self.min_date.day),
                                date=CURRENT_DATE,
                                style={
                                    'zIndex': 100,
                                    'background-color': '#2c3e50',
                                    'border': '2px solid #27ae60',
                                    'border-radius': '8px',
                                    'box-shadow': '0 4px 12px rgba(0,0,0,0.3)',
                                },
                            ),
                        ], 
                        className="p-3 rounded shadow-sm mb-4",
                        style={
                            'background': 'linear-gradient(135deg, #2c3e50 0%, #34495e 100%)',
                            'border': '2px solid #27ae60',
                            'box-shadow': '0 6px 20px rgba(0,0,0,0.4)'
                        },
                        direction='horizontal',
                        gap=3,
                ),
                
                # Timeline container with enhanced styling
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4(
                                "📺 Game Schedule Timeline", 
                                className="text-center text-light fw-bold mb-3",
                                style={
                                    'color': '#f8f9fa',
                                    'text-shadow': '2px 2px 4px rgba(0,0,0,0.5)'
                                }
                            ),
                            DashCalendarTimeline(
                                id='schedule',
                                defaultTimeStart=(self.df['start_date'].astype('int64') / 10**6).min(),
                                defaultTimeEnd=(self.df['start_date'].astype('int64') / 10**6).min() + 24 * 60 * 60 * 1000,
                                sidebarWidth=120,
                                lineHeight=65,
                                itemHeightRatio=1,
                                minZoom=8 * 60 * 60 * 1000,
                                disableScroll=False,
                                canMove=False,
                                canResize=False,
                                canChangeGroup=False,
                                sidebarHeaderVariant='left',
                                sidebarHeaderContent=html.H4(
                                    "📡 Network", 
                                    className='text-center text-light fw-bold',
                                    style={
                                        'color': '#f8f9fa',
                                        'background': 'linear-gradient(135deg, #27ae60 0%, #2ecc71 100%)',
                                        'padding': '10px',
                                        'margin': '0',
                                        'text-shadow': '1px 1px 2px rgba(0,0,0,0.3)'
                                    }
                                ),
                                timelineHeaderStyle={
                                    'background': 'transparent',
                                    'color': '#f8f9fa',
                                    'font-weight': 'bold'
                                },
                                customGroups=True,
                                customItems=True
                            ),
                        ], 
                        className="p-4 rounded shadow-lg",
                        style={
                            'background': 'linear-gradient(135deg, #1a252f 0%, #2c3e50 100%)',
                            'border': '2px solid #27ae60',
                            'box-shadow': '0 10px 30px rgba(0,0,0,0.5)',
                            'min-height': '600px'
                        })
                    ], width=12)
                ])
            ],fluid=True, className='pt-4 pb-4', style={'min-height': '100vh'})
        ], 
        fluid=True, 
        className='p-0',
        style={
            'background': 'linear-gradient(135deg, #0f1419 0%, #1a252f 50%, #2c3e50 100%)',
            'min-height': '100vh',
        })

    def _create_bins(self, df, n):
        """Create bins for data grouping."""
        x = round(len(df) / n)
        return np.array_split(df, x)

    def _get_team_data(self, game_row):
        """Extract team data for both teams efficiently using vectorized operations."""
        # Extract data for both teams at once
        team1_data = {
            'logo': game_row['logos'].iloc[1],
            'color': f"{game_row['color'].iloc[1]}",
            'location': game_row['team_name'].iloc[1],
            'alternate_color': f"{game_row['alternateColor'].iloc[1]}",
            'abbreviation': game_row['abbreviation'].iloc[1],
            'rank': game_row['rank'].iloc[1],
            'original_location': game_row['team_name'].iloc[1]
        }
        
        team2_data = {
            'logo': game_row['logos'].iloc[0],
            'color': f"{game_row['color'].iloc[0]}",
            'location': game_row['team_name'].iloc[0],
            'alternate_color': f"{game_row['alternateColor'].iloc[0]}",
            'abbreviation': game_row['abbreviation'].iloc[0],
            'rank': game_row['rank'].iloc[0],
            'original_location': game_row['team_name'].iloc[0]
        }
        
        # Add rank to location if ranked
        if team1_data['rank'] <= 25:
            team1_data['location'] = f"#{team1_data['rank']} {team1_data['location']}"
        if team2_data['rank'] <= 25:
            team2_data['location'] = f"#{team2_data['rank']} {team2_data['location']}"
            
        return team1_data, team2_data

    def _get_odds_data(self, game_row, team1_data, team2_data):
        """Extract and process odds information."""
        odds_details = game_row['formatted_spread'].iloc[0]
        over_under = game_row['over_under'].iloc[0]
        
        if pd.isna(odds_details) or odds_details == "":
            return "ODDS UNAVAILABLE", "#fff", "#000"
        
        odds_text = f"{odds_details} O/U {over_under}"
        
        # Determine odds team colors
        odds_team = str(odds_details).split("-")[0].strip()
        if odds_team == team1_data['original_location']:
            return odds_text, team1_data['color'], team1_data['alternate_color']
        elif odds_team == team2_data['original_location']:
            return odds_text, team2_data['color'], team2_data['alternate_color']
        else:
            return odds_text, "#fff", "#000"

    def _get_venue_data(self, game_row, team2_data):
        """Extract venue information and colors."""
        venue = f"At {game_row['venue'].iloc[0]}"
        
        if game_row['neutral_site'].iloc[0]:
            return venue, "#fff", "#000"
        else:
            return venue, team2_data['color'], team2_data['alternate_color']

    def _create_cache_key(self, game_data, dimensions):
        """Create a cache key for the scoreboard based on game data and dimensions."""
        # Create a string representation of the key data
        key_data = {
            'team1_name': game_data['team1']['location'],
            'team1_logo': game_data['team1']['logo'],
            'team1_text_color': game_data['team1']['alternate_color'],
            'team2_name': game_data['team2']['location'],
            'team2_logo': game_data['team2']['logo'],
            'team2_text_color': game_data['team2']['alternate_color'],
            'venue': game_data['venue']['text'],
            'odds': game_data['odds']['text'],
            'width': dimensions.get('width', 400) if dimensions else 400,
            'height': dimensions.get('height', 60) if dimensions else 60
        }
        
        # Create hash of the key data
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def _generate_scoreboard_svg_thread_safe(self, game_data, dimensions):
        """Thread-safe scoreboard generation with caching."""
        width = dimensions.get('width', 400) if dimensions else 400
        height = dimensions.get('height', 60) if dimensions else 60
        
        # Generate cache key
        cache_key = self._create_cache_key(game_data, dimensions)
        
        # Check cache first (thread-safe)
        with self._cache_lock:
            if cache_key in self._scoreboard_cache:
                return self._scoreboard_cache[cache_key]
        
        # Generate new scoreboard (this is the expensive operation)
        try:
            scoreboard = generate_custom_scoreboard(
                team1_name=game_data['team1']['location'],
                team1_logo_url=game_data['team1']['logo'],
                team1_text_color=game_data['team1']['alternate_color'],
                team2_name=game_data['team2']['location'],
                team2_logo_url=game_data['team2']['logo'],
                team2_text_color=game_data['team2']['alternate_color'],
                venue=game_data['venue']['text'],
                venue_text_color=game_data['venue']['text_color'],
                score_line=game_data['odds']['text'],
                score_text_color=game_data['odds']['text_color'],
                team1_name_bg_color=game_data['team1']['color'],
                team1_logo_bg_color=game_data['team1']['color'],
                team2_name_bg_color=game_data['team2']['color'],
                team2_logo_bg_color=game_data['team2']['color'],
                venue_bg_color=game_data['venue']['bg_color'],
                score_bg_color=game_data['odds']['bg_color'],
                width=width,
                height=height
            )
            
            svg_string = scoreboard.as_svg()
            svg_base64 = base64.b64encode(svg_string.encode()).decode()
            
            # Cache the result (thread-safe)
            with self._cache_lock:
                self._scoreboard_cache[cache_key] = svg_base64
            
            return svg_base64
            
        except Exception as e:
            # Return a simple error placeholder if generation fails
            print(f"Error generating scoreboard: {e}")
            return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjYwIj48dGV4dCB4PSIxMCIgeT0iMzAiPkVycm9yPC90ZXh0Pjwvc3ZnPg=="

    def _process_single_game(self, game_data_tuple):
        """Process a single game and return the item and scoreboard content."""
        game, group_id, dimensions, game_idx = game_data_tuple
        
        try:
            game = game.reset_index(drop=True)
            
            # Create game data cache key for processed game data
            game_cache_key = f"{game['id'].iloc[0]}_{group_id}"
            
            # Check if game data is already processed
            with self._game_data_lock:
                if game_cache_key in self._game_data_cache:
                    processed_game_data = self._game_data_cache[game_cache_key]
                else:
                    # Extract team data efficiently
                    team1_data, team2_data = self._get_team_data(game)
                    
                    # Extract odds and venue data
                    odds_text, odds_bg, odds_text_color = self._get_odds_data(game, team1_data, team2_data)
                    venue_text, venue_bg, venue_text_color = self._get_venue_data(game, team2_data)
                    
                    # Create game data structure
                    processed_game_data = {
                        'game_id': game['id'].iloc[0],
                        'start_time_ms': game['start_time_ms'].iloc[0],
                        'end_time_ms': game['end_time_ms'].iloc[0],
                        'team1': team1_data,
                        'team2': team2_data,
                        'odds': {'text': odds_text, 'bg_color': odds_bg, 'text_color': odds_text_color},
                        'venue': {'text': venue_text, 'bg_color': venue_bg, 'text_color': venue_text_color}
                    }
                    
                    # Cache processed game data
                    self._game_data_cache[game_cache_key] = processed_game_data
            
            # Create timeline item
            item = {
                'id': processed_game_data['game_id'],
                'group': group_id,
                'title': f"{processed_game_data['team1']['location']} vs {processed_game_data['team2']['location']}\n{processed_game_data['venue']['text']}\n{processed_game_data['odds']['text']}",
                'start_time': processed_game_data['start_time_ms'],
                'end_time': processed_game_data['end_time_ms'],
                'itemProps': {'style': {'background': 'transparent'}}
            }
            
            # Generate scoreboard SVG (this is the expensive operation)
            svg_base64 = self._generate_scoreboard_svg_thread_safe(processed_game_data, dimensions)
            
            # Create scoreboard content
            content = html.Img(
                src=f"data:image/svg+xml;base64,{svg_base64}",
                style={'width': '100%', 'height': 'auto', 'vertical-align': 'baseline'}
            )
            
            return game_idx, item, content
            
        except Exception as e:
            print(f"Error processing game {game_idx}: {e}")
            # Return minimal data in case of error
            return game_idx, {
                'id': f'error_{game_idx}',
                'group': group_id,
                'title': 'Error loading game',
                'start_time': 0,
                'end_time': 0,
                'itemProps': {'style': {'background': 'red'}}
            }, html.Div("Error")

    def _process_games_parallel(self, games_data, dimensions):
        """Process multiple games in parallel using ThreadPoolExecutor."""
        if not games_data:
            return [], []
        
        # Prepare data for parallel processing
        game_tuples = []
        for game_idx, (game, group_id) in enumerate(games_data):
            game_tuples.append((game, group_id, dimensions, game_idx))
        
        # Process games in parallel
        items = [None] * len(game_tuples)
        custom_items_content = [None] * len(game_tuples)
        
        # Use ThreadPoolExecutor for parallel processing
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            for game_tuple in game_tuples:
                future = executor.submit(self._process_single_game, game_tuple)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    game_idx, item, content = future.result(timeout=30)  # 30 second timeout
                    items[game_idx] = item
                    custom_items_content[game_idx] = content
                except Exception as e:
                    print(f"Error in parallel processing: {e}")
                    # Handle failed tasks gracefully
                    continue
        
        # Filter out None values (failed tasks)
        items = [item for item in items if item is not None]
        custom_items_content = [content for content in custom_items_content if content is not None]
        
        return items, custom_items_content

    def _get_cached_schedule_data(self, date_value: str, dimensions: Dict) -> Optional[Tuple]:
        """Get cached schedule data for a specific date."""
        # Create cache key based on date and dimensions
        cache_key = f"schedule_data_{date_value}_{hash(str(dimensions))}"
        return self.server_cache.get(cache_key)

    def _set_cached_schedule_data(self, date_value: str, dimensions: Dict, schedule_data: Tuple) -> None:
        """Cache schedule data for a specific date."""
        cache_key = f"schedule_data_{date_value}_{hash(str(dimensions))}"
        self.server_cache.set(cache_key, schedule_data)

    def _setup_callbacks(self):
        """Setup application callbacks."""
        @self.app.callback(
            [
                Output('schedule', 'groups'),
                Output('schedule', 'items'),
                Output('schedule', 'visibleTimeStart'),
                Output('schedule', 'visibleTimeEnd'),
                Output('schedule', 'customGroupsContent'),
                Output('schedule', 'customItemsContent'),
                Output('schedule', 'itemsStyle'),
                Output('schedule', 'groupsStyle')
            ],
            [
                Input('schedule', 'itemClickData'),
                Input('schedule', 'itemDimensions'),
                Input('date-picker', 'date')
            ]
        )
        def update_schedule(click_data, dimensions, date_value):
            start_time = time.time()
            
            # Try to get cached data first
            cached_data = self._get_cached_schedule_data(date_value, dimensions or {})
            if cached_data is not None:
                print(f"Schedule data loaded from cache for date: {date_value}")
                return cached_data
            
            print(f"Processing schedule data for date: {date_value}")
            
            # Filter data by selected date
            df_filtered = self.df[self.df['start_date'].dt.date == pd.to_datetime(date_value[:10]).date()].copy()
            
            # Early return if no data
            if df_filtered.empty:
                empty_result = ([], [], 0, 0, [], [], {}, {})
                self._set_cached_schedule_data(date_value, dimensions or {}, empty_result)
                return empty_result
            
            # Initialize output lists
            groups = []
            custom_groups_content = []
            
            # Prepare games data for parallel processing
            games_data = []
            
            # Process each unique station
            stations = df_filtered['outlet'].unique()
            
            for group_id, station in enumerate(stations):
                station_data = df_filtered[df_filtered['outlet'] == station]
                df_split = self._create_bins(station_data, 2)
                
                # Add games to parallel processing list
                for game in df_split:
                    games_data.append((game, group_id))
                
                # Create group
                groups.append({
                    'id': group_id,
                    'title': station,
                    'stackItems': True
                })
                
                # Create group content (network logo)
                network_logo = station_data['outlet_logos'].iloc[0]
                custom_groups_content.append(
                    html.Div([
                        html.Img(
                            src=network_logo,
                            style={"max-width": '100%', 'max-height': '100%'}
                        )
                    ], style={'width': '100%', 'height': '100%'}, className='text-center')
                )
            
            # Process all games in parallel
            items, custom_items_content = self._process_games_parallel(games_data, dimensions)
            
            # Define styles
            custom_styles = {'height': '100%', 'width': '100%'}
            
            # Prepare result tuple
            result = [
                groups, items,
                df_filtered['start_time_ms'].min(),
                df_filtered['end_time_ms'].max(),
                custom_groups_content, custom_items_content,
                custom_styles, custom_styles
            ]
            
            # Cache the result for future requests
            self._set_cached_schedule_data(date_value, dimensions or {}, result)
            
            end_time = time.time()
            print(f"✓ Schedule update completed in {end_time - start_time:.2f} seconds with {len(games_data)} games (CACHED for future users)")
            
            return result

    def run(self, debug=True):
        """Run the application."""
        self.app.run(debug=debug)

    def clear_cache(self):
        """Clear all caches if needed."""
        with self._cache_lock:
            self._scoreboard_cache.clear()
        with self._game_data_lock:
            self._game_data_cache.clear()
        self.server_cache.clear()

    def shutdown(self):
        """Clean shutdown of thread pool."""
        self.executor.shutdown(wait=True)

    def __del__(self):
        """Ensure proper cleanup."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Create and run the application
if __name__ == '__main__':
    # Configuration options for caching
    CACHE_CONFIG = {
        'cache_type': CACHE_TYPE,  # "file", "redis", or "memory"
        'cache_ttl': CACHE_TTL,    # Cache TTL in seconds
        'max_workers': 4           # Number of worker threads
    }
    
    print(f"Starting CFB App with {CACHE_CONFIG['cache_type'].upper()} caching (TTL: {CACHE_CONFIG['cache_ttl']}s)")
    
    cfb_app = CFBGuideApp(
        max_workers=CACHE_CONFIG['max_workers'],
        cache_type=CACHE_CONFIG['cache_type'],
        cache_ttl=CACHE_CONFIG['cache_ttl']
    )
    
    try:
        cfb_app.run(host='0.0.0.0', port=8050, debug=False)
    finally:
        cfb_app.shutdown()