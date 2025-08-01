# HackTheDelay

🛠️ Feature Engineering Summary
We performed feature engineering on raw GPS and timetable data to extract meaningful insights for delay detection:

Distance to Nearest Stop (DistanceToNearestStop_m): Calculated geodesic distance from each GPS point to the closest scheduled stop using Haversine formula via geopy.

Time Difference to Stop (TimeDiffToStop_min): Computed the time offset between the actual GPS timestamp and the scheduled time at the nearest stop.

Stopped Duration (StoppedDuration): Grouped consecutive GPS points with speed = 0 to identify stop events and calculated how long the vehicle remained stopped.

Temporal Features: Extracted Hour, DayOfWeek, and Is_Night from timestamps to capture time-based behavioral patterns.

These engineered features will support downstream tasks like delay classification and route performance diagnostics.

