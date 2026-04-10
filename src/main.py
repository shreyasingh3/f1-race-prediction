import os
import fastf1
import pandas as pd

#cache folder
cache_folder = "cache_folder"
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

fastf1.Cache.enable_cache(cache_folder)

#Monaco GP 2024
session = fastf1.get_session(2024, "Monaco", "R")
session.load()


laps = session.laps

df = laps[["Driver", "LapTime", "Compound"]]

print(df)