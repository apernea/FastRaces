import fastf1
import pandas as pd
import numpy as np
import time

fastf1.Cache.enable_cache('cache')  # Enable caching to speed up data retrieval

def get_race_history(year):
    """Fetch race history for a given year."""

    driver_history = {}
    team_history = {}

    try:
        schedule = fastf1.get_event_schedule(year)
        
        races = schedule[schedule['EventFormat'] == 'conventional']   
    except:
        return None
    
    for _, row in races.iterrows():
        round_num = row['RoundNumber']
        event_name = row['EventName']

        try:
            race = fastf1.get_session(year, round_num, 'R')
            race.load(laps=False, telemetry=False, weather=False)

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
            time.sleep(5) # To avoid API limits

            quali = fastf1.get_session(current_year, round_num, 'Q')
            quali.load(laps=False, telemetry=False, weather=False)

            race = fastf1.get_session(current_year, round_num, 'R')
            race.load(laps=False, telemetry=False, weather=False)

            weather = race.weather_data
            air_temp = weather['AirTemp'].mean()
            track_temp = weather['TrackTemp'].mean()
            humidity = weather['Humidity'].mean()

            if not weather.empty and 'Rainfall' in weather.columns:
                rainfall = weather['Rainfall'].any()
            else:
                rainfall = False

            wind_speed = weather['WindSpeed'].mean()
            wind_direction = weather['WindDirection'].mean()

            pole_lap = quali.pick_fastest()
            pole_time = pole_lap['LapTime'].total_seconds()

            for driver in race.results['Abbreviation']:
                try:
                    if driver not in quali.results.index:
                        continue

                    q_res = quali.results.loc[driver]
                    grid_pos = q_res['GridPosition']
                    team_name = q_res['TeamName']

                    best_quali_lap = q_res['Q3'] if not pd.isna(q_res['Q3']) else (q_res['Q2'] if not pd.isna(q_res['Q2']) else q_res['Q1'])

                    if pd.isnull(best_quali_lap):
                        time_delta = 5.0
                    else:
                        time_delta = best_quali_lap.total_seconds() - pole_time

                    if(event_name, driver) in driver_history:
                        prev_driver_pos = driver_history[(event_name, driver)]
                        is_rookie = 0
                    elif (event_name, team_name) in team_history:
                        prev_driver_pos = team_history[(event_name, team_name)]
                        is_rookie = 1
                    else:
                        prev_driver_pos = 20
                        is_rookie = 1
                        
                    r_res = race.results.loc[driver]

                    driver_laps = race.laps.pick_driver(driver)
                    # Count how many times they entered the pits
                    pitstops = driver_laps['PitInTime'].notna().sum()
                    status = r_res['Status']
                    points = r_res['Points']

                    final_pos = r_res['ClassifiedPosition']
                    try:
                        final_pos = float(final_pos)
                    except:
                        pass

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
                        'Race_Rainfall': int(rainfall), # Convert True/False to 1/0
                        'Race_WindSpeed': wind_speed,
                        'Race_WindDirection': wind_direction,
                        'Pitstops': pitstops,
                        'Status': status,
                        'Points': points,  
                        'Final_Pos': final_pos
                    })

                except Exception:
                    continue 

        except ValueError:
            print("!! RATE LIMIT HIT !! Pausing for 60 seconds...")
            time.sleep(60)
        except Exception as e:
            print(f"Skipping {event_name}: {e}")

    return pd.DataFrame(season_data)


years_to_process = [2025]

all_data = []

for year in years_to_process:
    df_year = process_season_data(year)
    all_data.append(df_year)

final_dataset = pd.concat(all_data, ignore_index=True)

# Save
final_dataset.to_csv('f1_race_data_w_weather.csv', index=False)
print("\nFull Dataset with Weather Created Successfully!")
print(final_dataset.head())