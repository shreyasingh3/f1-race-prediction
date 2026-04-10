import os
import fastf1
import pandas as pd
import numpy as np

cache_path = "cache_folder"
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

# enable FastF1 Cacheing
fastf1.Cache.enable_cache(cache_path)

# Load 2024 AUS F1 GP Data
session_2024 = fastf1.get_session(2024, "Monaco", "R")
session_2024.load()

print("Session Loaded")

laps = session_2024.laps
df = laps[["Driver", "LapTime", "Compound"]].copy()

print(df.head())