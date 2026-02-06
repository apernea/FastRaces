import fastf1
from fastf1.req import RateLimitExceededError
import pandas as pd
import numpy as np
import time
import config 

fastf1.Cache.enable_cache('cache')  # Enable caching to speed up data retrieval

def get_race_history(year):
    """Fetch race history for a given year."""

    driver_history = {}
    team_history = {}

    try:
        schedule = fastf1.get_event_schedule(year)
        
        races = schedule[schedule['EventFormat'] == 'conventional']   
    except:
        return None, None
    
    for _, row in races.iterrows():
        round_num = row['RoundNumber']
        event_name = row['EventName']

        try:
            race = fastf1.get_session(year, round_num, 'R')
            race.load(laps=False, telemetry=False, weather=False, messages=False)
            race_team_stats = {}

            for driver in race.results['Abbreviation']:
                try:
                    drv_res = race.results.loc[driver]
                    team_name = drv_res['TeamName']
                    position = drv_res['ClassifiedPosition']

                    try:
                        position = float(position)
                    except:
                        continue

                    driver_history[(event_name, driver)] = position

                    if team_name not in race_team_stats:
                        race_team_stats[team_name] = []
                    race_team_stats[team_name].append(position)
                except:
                    continue
            for team, positions in race_team_stats.items():
                if(len(positions) > 0):
                    avg_position = np.sum(positions) / len(positions)
                    team_history[(event_name, team)] = avg_position
        except Exception as e:
            print(f"Error fetching data for {year} Round {round_num}: {e}")
            continue

    return driver_history, team_history

def process_season_data(current_year):
    """Process race data for a given season using history data and current data."""

    driver_history, team_history = get_race_history(current_year - 1)

    race_schedule = fastf1.get_event_schedule(current_year)
    races = race_schedule[race_schedule['EventFormat'] == 'conventional']

    season_data = []

    for _, row in races.iterrows():
        round_num = row['RoundNumber']
        event_name = row['EventName']

        try:
            time.sleep(5)  # To avoid API limits

            quali = fastf1.get_session(current_year, round_num, 'Q')
            quali.load(laps=True, telemetry=False, weather=False, messages=False)
            
            quali_by_driver = quali.results.set_index('Abbreviation')

            race = fastf1.get_session(current_year, round_num, 'R')
            race.load(laps=True, telemetry=False, weather=True, messages=False) 
            
            race_by_driver = race.results.set_index('Abbreviation')

            # Weather data
            weather = race.weather_data
            air_temp = weather['AirTemp'].mean()
            track_temp = weather['TrackTemp'].mean()
            humidity = weather['Humidity'].mean()
            rainfall = weather['Rainfall'].any() if 'Rainfall' in weather.columns else False
            wind_speed = weather['WindSpeed'].mean()
            wind_direction = weather['WindDirection'].mean()

            # Pole time
            pole_lap = quali.laps.pick_fastest()
            pole_time = pole_lap['LapTime'].total_seconds()

            # Process each driver
            for driver in race.results['Abbreviation']:
                try:
                    print(f"Processing driver: {driver}")

                    # Check if driver was in qualifying
                    if driver not in quali_by_driver.index:
                        print(f"  {driver} not in qualifying")
                        continue

                    # Get qualifying data
                    q_res = quali_by_driver.loc[driver]
                    grid_pos = q_res['GridPosition']
                    team_name = q_res['TeamName']

                    # Best quali lap time
                    best_quali_lap = q_res['Q3'] if not pd.isna(q_res['Q3']) else (
                        q_res['Q2'] if not pd.isna(q_res['Q2']) else q_res['Q1']
                    )

                    if pd.isnull(best_quali_lap):
                        time_delta = 5.0
                    else:
                        time_delta = best_quali_lap.total_seconds() - pole_time

                    # Historical data
                    if (event_name, driver) in driver_history:
                        prev_driver_pos = driver_history[(event_name, driver)]
                        is_rookie = 0
                    elif (event_name, team_name) in team_history:
                        prev_driver_pos = team_history[(event_name, team_name)]
                        is_rookie = 1
                    else:
                        prev_driver_pos = 20
                        is_rookie = 1
                    
                    # Get race data
                    r_res = race_by_driver.loc[driver]
                    
                    # Pitstops
                    driver_laps = race.laps.pick_drivers(driver)
                    pitstops = driver_laps['PitInTime'].notna().sum() if not driver_laps.empty else 0
                    
                    # Race results
                    status = r_res['Status']
                    points = r_res['Points']
                    final_pos = r_res['ClassifiedPosition']
                    
                    try:
                        final_pos = float(final_pos)
                    except:
                        continue  # Skip if position can't be converted

                    print(f"  Final position: {final_pos}")

                    # Append data
                    season_data.append({
                        'Season': current_year,
                        'Round': round_num,
                        'Event': event_name,
                        'Driver': driver,
                        'Team': team_name,
                        'Grid_Pos': grid_pos,
                        'Quali_Delta': time_delta,
                        'Prev_Year_Race_Pos': prev_driver_pos,
                        'Is_Rookie': is_rookie,
                        'Race_AirTemp': air_temp,
                        'Race_TrackTemp': track_temp,
                        'Race_Humidity': humidity,
                        'Race_Rainfall': int(rainfall),
                        'Race_WindSpeed': wind_speed,
                        'Race_WindDirection': wind_direction,
                        'Pitstops': pitstops,
                        'Status': status,
                        'Points': points,  
                        'Final_Pos': final_pos
                    })
                    
                    print(f"Added {driver} to season_data")

                except Exception as e:
                    print(f"Error with {driver}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue 

        except RateLimitExceededError as e:
            print(f"Error processing {event_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return pd.DataFrame(season_data)

def fetch_all_data():
    """
    Main function to orchestrate the data fetching process.
    """
    all_data = []
    for year in config.YEARS_TO_PROCESS:
        df_year = process_season_data(year)
        all_data.append(df_year)

    if all_data:
        final_dataset = pd.concat(all_data, ignore_index=True)
        final_dataset.to_csv(config.RAW_DATA_PATH, index=False)
        print(f"\nComprehensive Dataset saved to {config.RAW_DATA_PATH}")
        print(f"Total Rows: {len(final_dataset)}")
    else:
        print("No data was collected.")

if __name__ == '__main__':
    # This allows you to run this script directly if you only want to fetch data
    fetch_all_data()