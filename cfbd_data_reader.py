import os
import cfbd
from cfbd.rest import ApiException
import pandas as pd
import re
from dotenv import load_dotenv

load_dotenv()


configuration = cfbd.Configuration(access_token = os.getenv('CFBD_API'))

api_instance1 = cfbd.GamesApi(cfbd.ApiClient(configuration))
api_instance2 = cfbd.TeamsApi(cfbd.ApiClient(configuration))
api_instance3 = cfbd.BettingApi(cfbd.ApiClient(configuration))
api_instance4 = cfbd.RankingsApi(cfbd.ApiClient(configuration))


def add_outlet_logos(df, base_logo_path="assets/tv-logos/"):
    # Create the new column 'outlet_logos'
    df['outlet_logos'] = df['outlet'].apply(lambda outlet: find_logo(outlet, base_logo_path))
    return df

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

def find_logo(outlet_name, base_path):
    folder_path = f"{base_path}{outlet_name} tv channel logo/"
    
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                return os.path.join(folder_path, file)

    return None

def handle_duplicate_ids(df):

    df_copy = df.copy()
    
    id_pair_counts = {}
  
    new_ids = []
    
    i = 0
    while i < len(df_copy):
        current_id = str(df_copy.iloc[i]['id'])
       
        if current_id in id_pair_counts:
            id_pair_counts[current_id] += 1

            new_id = f"{current_id}_{id_pair_counts[current_id]}"
        else:
            id_pair_counts[current_id] = 0
            new_id = current_id
        
        new_ids.append(new_id)

        if i + 1 < len(df_copy) and str(df_copy.iloc[i + 1]['id']) == current_id:
            new_ids.append(new_id)
            i += 2  
        else:
            i += 1
    
    df_copy['id'] = new_ids
    
    return df_copy

def split_home_away(df):
    rows = []
    for _, row in df.iterrows():
        home_row = {
            'id': row['id'],
            'team_type': 'home',
            'team_id': row['homeId'],
            'team_name': row['homeTeam'],
            'team_conference': row['homeConference'],
            'start_date': row['startDate'],
            'venue': row['venue'],
            'neutral_site': row['neutralSite'],
            'outlet': row['outlet']
        }
        away_row = {
            'id': row['id'],
            'team_type': 'away',
            'team_id': row['awayId'],
            'team_name': row['awayTeam'],
            'team_conference': row['awayConference'],
            'start_date': row['startDate'],
            'venue': row['venue'],
            'neutral_site': row['neutralSite'],
            'outlet': row['outlet']
        }
        rows.append(home_row)
        rows.append(away_row)
    
    return pd.DataFrame(rows)


def read_data():
    try:
        api_response_1 = api_instance1.get_games(year=2024)
        df1 = pd.DataFrame([game.to_dict() for game in api_response_1],columns=['id','startDate','neutralSite','venue','homeId','homeTeam','awayId','awayTeam','homeConference','awayConference'])
    
        api_response_2 = api_instance1.get_media(year=2024)
        df2 = pd.DataFrame([game.to_dict() for game in api_response_2],columns=['id','outlet'])

        df2['outlet'] = df2['outlet'].apply(sanitize_filename)
    
        df_init1 = pd.merge(df1, df2, on="id")
    
        modified_data = split_home_away(df_init1)
    
        api_response_3 = api_instance2.get_teams()
        
        df3 = pd.DataFrame([game.to_dict() for game in api_response_3],columns=['id','abbreviation','color','alternateColor','logos'])
    
        df3['logos'] = [x[0] if isinstance(x, list) and len(x) > 0 else '' for x in df3['logos']]
    
        df3.rename(columns={'id':'team_id'},inplace=True)
    
        df_init2 = pd.merge(modified_data,df3, on="team_id")
    
        api_response_4 = api_instance3.get_lines(year=2024)
        records = []
        for game in api_response_4:
            if game.to_dict()['lines']:
                last_line = game.to_dict()['lines'][-1]
                record = {
                    'id': game.to_dict()['id'],
                    'formatted_spread': last_line['formattedSpread'],
                    'over_under': last_line['overUnder']
                }
                records.append(record)
    
        df4 = pd.DataFrame(records)
    
        df_init3 = pd.merge(df_init2,df4,on="id")
    
        add_outlet_logos(df_init3)

        api_response_5 = api_instance4.get_rankings(year=2024)

        ranks_schools = []
        for item in api_response_5:
            for poll in item.to_dict()['polls']:
                for rank_info in poll['ranks']:
                    ranks_schools.append({
                        'rank': rank_info['rank'],
                        'school': rank_info['school']
                    })

        df5 = pd.DataFrame(ranks_schools)

        df5.rename(columns={'school':'team_name'},inplace=True)

        school_rank_map = dict(zip(df5['team_name'], df5['rank']))

        df_init3['rank'] = df_init3['team_name'].map(school_rank_map)
        
        df_final = handle_duplicate_ids(df_init3)

        return df_final
    
    except ApiException as e:
        print("Exception: %s\n" % e)