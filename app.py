import pandas as pd
import numpy as np
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
from dash_calendar_timeline import DashCalendarTimeline
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock

from banner import generate_custom_scoreboard
import base64
# from cfbd_data_reader import read_data
from datetime import datetime

# Get current date
CURRENT_DATE = datetime.now().date()


class CFBGuideApp:
    def __init__(self, max_workers=8):
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=0.3"}],
            url_base_pathname='/college-football-schedule/'
        )
        self.app.title = "College Football"
        self.server = self.app.server
        self.df = self._load_and_prepare_data()
        self.min_date, self.max_date = self._get_date_bounds()
        self.available_dates = self._get_available_dates()
        
        # Threading configuration
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Thread-safe cache for SVG scoreboards
        self._scoreboard_cache = {}
        self._cache_lock = Lock()
        
        # Pre-computation cache for game data processing
        self._game_data_cache = {}
        self._game_data_lock = Lock()
        
        # Store dimensions for reuse
        self._cached_dimensions = {}
        self._dimensions_lock = Lock()
        
        self._setup_layout()
        self._setup_callbacks()

    def _load_and_prepare_data(self):
        """Load and prepare the data with optimized column operations."""
        # df = read_data() # Read data from API
        df = pd.read_csv('./data/cfbd_data.csv') # Read data from local file
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = df['start_date'] + pd.Timedelta(hours=3, minutes=30)
        
        # Pre-compute timestamp conversions
        df['start_time_ms'] = df['start_date'].astype("int64") // 10**6
        df['end_time_ms'] = df['end_date'].astype("int64") // 10**6
        
        return df

    def _get_date_bounds(self):
        """Get min and max dates from the dataset."""
        date_col = self.df['start_date']
        return date_col.min(), date_col.max()
    
    def _get_available_dates(self):
        """Get list of available dates from the dataset."""
        
        # Get unique dates from the dataset
        available_dates = sorted(self.df['start_date'].dt.date.unique())
        return available_dates

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
            
            # Hidden store for dimensions
            dcc.Store(id='dimensions-store', storage_type='memory'),
            
            # Main content container with dark theme
            dbc.Container([
                # Date picker with enhanced styling and loading spinner
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
                        date=min(self.available_dates, key=lambda d: abs(d - CURRENT_DATE)),
                        disabled_days=[date for date in pd.date_range(self.min_date, self.max_date).date 
                                     if date not in self.available_dates],
                        style={
                            'zIndex': 100,
                            'background-color': '#2c3e50',
                            'border': '2px solid #27ae60',
                            'border-radius': '8px',
                            'box-shadow': '0 4px 12px rgba(0,0,0,0.3)',
                        },
                    ),
                    # Loading spinner
                    dbc.Spinner(
                        id="loading-spinner",
                        color="success"
                    ),

                    html.Span("ℹ", id="tooltip-target",
                        className="fs-1 fw-bold mb-0 ms-auto",
                        style={'color': '#188754', 'cursor': 'pointer', 'padding-left': '10px'}
                    ),
                    dbc.Tooltip(
                        [
                            html.P("1. Select a date to view the game schedule. Dates without games are disabled.",className='m-0'),
                            html.Br(),
                            html.P("2. Games are grouped by their broadcasting network.",className='m-0'),
                            html.Br(),
                            html.P("3. Use left and right arrow keys to scroll across the timeline or use right-click to hold and drag the timeline.",className='m-0'),
                            html.Br(),
                            html.P("4. Zoom in and out using the mouse wheel or pinch gestures.",className='m-0'),
                        ],
                        target="tooltip-target",
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
                                minZoom=10 * 60 * 60 * 1000,
                                canMove=False,
                                canResize=False,
                                canChangeGroup=False,
                                sidebarHeaderVariant='left',
                                dateHeaderLabelFormat='HH:00',
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

    def _setup_callbacks(self):
        """Setup application callbacks."""

        # Separate callback to capture dimensions only on first load
        @self.app.callback(
            Output('dimensions-store', 'data'),
            [Input('schedule', 'itemDimensions')],
            [State('dimensions-store', 'data')],
            prevent_initial_call=False
        )
        def update_dimensions_store(item_dimensions, current_dimensions):
            """Update dimensions store only if we have new dimensions and they're different."""
            if item_dimensions and item_dimensions != current_dimensions:
                # print(f"Dimensions updated: {item_dimensions}")
                # Update cached dimensions
                with self._dimensions_lock:
                    self._cached_dimensions = item_dimensions
                return item_dimensions
            
            # Return current dimensions to prevent unnecessary updates
            return current_dimensions

        # Loading spinner callback
        @self.app.callback(
            Output("loading-spinner", "spinner_style", allow_duplicate=True),
            [Input('date-picker', 'date')],
            prevent_initial_call='initial_duplicate',
        )
        def show_loading_spinner(date_value):
            """Show/hide loading spinner based on callback execution."""
            if date_value:
                return {"display": "block"}
            return {"display": "none"}
        
        # Main schedule update callback - now uses State for dimensions instead of Input
        @self.app.callback(
            [
                Output('schedule', 'groups'),
                Output('schedule', 'items'),
                Output('schedule', 'visibleTimeStart'),
                Output('schedule', 'visibleTimeEnd'),
                Output('schedule', 'maxZoom'),
                Output('schedule', 'customGroupsContent'),
                Output('schedule', 'customItemsContent'),
                Output('schedule', 'itemsStyle'),
                Output('schedule', 'groupsStyle'),
                Output('loading-spinner', 'spinner_style')
            ],
            [
                Input('date-picker', 'date'),
                Input('dimensions-store', 'data')
            ]
        )
        def update_schedule(date_value, stored_dimensions):
            start_time = time.time()
            
            # Use stored dimensions or fallback to cached/default dimensions
            dimensions = stored_dimensions or self._cached_dimensions
            
            # Filter data by selected date
            df_filtered = self.df[self.df['start_date'].dt.date == pd.to_datetime(date_value[:10]).date()].copy()
            
            # Early return if no data
            if df_filtered.empty:
                return [], [], 0, 0, [], [], {}, {}, {'display': 'none'}
            
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
            
            end_time = time.time()
            print(f"Schedule update completed in {end_time - start_time:.2f} seconds with {len(games_data)} games")
            
            return [
                groups, items,
                df_filtered['start_time_ms'].min(),
                df_filtered['end_time_ms'].max(),
                df_filtered['end_time_ms'].max() - df_filtered['start_time_ms'].min(),
                custom_groups_content, custom_items_content,
                custom_styles, custom_styles, {'display': 'none'}
            ]

    def run(self, debug=False):
        """Run the application."""
        self.app.run(debug=debug)

    def clear_cache(self):
        """Clear all caches if needed."""
        with self._cache_lock:
            self._scoreboard_cache.clear()
        with self._game_data_lock:
            self._game_data_cache.clear()

    def shutdown(self):
        """Clean shutdown of thread pool."""
        self.executor.shutdown(wait=True)

    def __del__(self):
        """Ensure proper cleanup."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Create and run the application
if __name__ == '__main__':
    cfb_app = CFBGuideApp(max_workers=8)  # Adjust max_workers based on your CPU cores
    try:
        cfb_app.run()
    finally:
        cfb_app.shutdown()