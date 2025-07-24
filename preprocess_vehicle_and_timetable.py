import pandas as pd
from datetime import datetime
from geopy.distance import geodesic

# === STEP 1: Preprocess Timetable Files === #

def clean_and_save_timetable(file_path):
    # Load timetable
    df = pd.read_csv(file_path)
    
    # Assign proper column names
    columns = ['SNo', 'Location', 'Primary', 'BoardingPoint', 'Time', 'Day', 'Latitude', 'Longitude', 'PointType']
    df.columns = columns
    
    # Convert 'Time' to proper format
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce').dt.time
    
    # Drop invalid rows
    df.dropna(subset=['Time', 'Latitude', 'Longitude'], inplace=True)
    
    # Save cleaned timetable
    df.to_csv(file_path, index=False)
    return df

# Clean and save both timetables
hyd_pune = clean_and_save_timetable("timetable_hyd_pune.csv")
pune_hyd = clean_and_save_timetable("timetable_pune_hyd.csv")

# === STEP 2: Feature Engineering on Vehicle GPS Data === #

def process_vehicle_data(vehicle_path, timetable_df, output_path):
    # Load vehicle data
    vehicle_df = pd.read_csv(vehicle_path)
    vehicle_df['Timestamp'] = pd.to_datetime(vehicle_df['Timestamp'])

    # Ensure timetable time is datetime.time
    timetable_df['Time'] = pd.to_datetime(timetable_df['Time'], errors='coerce').dt.time

    # Function to compute nearest stop distance and time difference
    def get_nearest_stop_info(lat, lon, timestamp):
        min_dist = float('inf')
        closest_time_diff = None
        for _, row in timetable_df.iterrows():
            stop_loc = (row['Latitude'], row['Longitude'])
            current_loc = (lat, lon)
            dist = geodesic(current_loc, stop_loc).meters
            if dist < min_dist:
                min_dist = dist
                try:
                    stop_time = datetime.combine(timestamp.date(), row['Time'])
                    closest_time_diff = (timestamp - stop_time).total_seconds() / 60  # in minutes
                except:
                    closest_time_diff = None
        return min_dist, closest_time_diff

    # Apply function to each row
    vehicle_df[['DistanceToNearestStop_m', 'TimeDiffToStop_min']] = vehicle_df.apply(
        lambda row: pd.Series(get_nearest_stop_info(
            row['Latitude'], row['Longitude'], row['Timestamp']
        )), axis=1
    )

    # Detect stopped durations (Speed = 0)
    vehicle_df['IsStopped'] = vehicle_df['Speed'] == 0
    vehicle_df['StopGroup'] = (vehicle_df['IsStopped'] != vehicle_df['IsStopped'].shift()).cumsum()
    vehicle_df['StoppedDuration'] = vehicle_df.groupby('StopGroup')['Timestamp'].transform(
        lambda x: (x.max() - x.min()).total_seconds() / 60
    )
    vehicle_df['StoppedDuration'] = vehicle_df['StoppedDuration'].where(vehicle_df['IsStopped'], 0)

    # Save final output
    vehicle_df.to_csv(output_path, index=False)
    print(f"[INFO] Processed data saved to {output_path}")

# Run processing
process_vehicle_data("vehicle.csv", hyd_pune, "vehicle_with_features.csv")
