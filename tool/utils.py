import os
import multiprocessing as mp
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import aiohttp
from dotenv import load_dotenv
from urllib.parse import urlencode

from tool.path import BASE_DIR

load_dotenv(BASE_DIR / ".env")

API_KEY = os.environ["API_KEY"]
ENDPOINT_URL = os.environ["ENDPOINT_URL"]

def add_features(data: pd.DataFrame) -> pd.DataFrame:

    data["rolling_mean"] = data.groupby("city")["temperature"].transform(
        lambda group: group.rolling(window=30, min_periods=1).mean()
    )
    data["season_mean"] = data.groupby(["city", "season"])["temperature"].transform(
        lambda group: group.mean())
    data["season_dev"] = data.groupby(["city", "season"])["temperature"].transform(
        lambda group: group.std())
    data["outlier"] = (
        np.abs(data["temperature"] - data["season_mean"]) > 2 * data["season_dev"])
    return data

season_by_month = {
    1: "winter", 2: "winter", 3: "spring",  4: "spring", 5: "spring", 6: "summer", 7: "summer",
    8: "summer", 9: "autumn", 10: "autumn", 11: "autumn", 12: "winter",
}

def check_temperature(
    data: pd.DataFrame,
    city: str,
    current_temperature: float,
) -> bool:

    current_month = datetime.now().month
    current_season = season_by_month[current_month]

    data_city_season = data.query("city == @city and season == @current_season")

    if data_city_season.empty:
        return False 

    mean_temperature = data_city_season["season_mean"].iloc[0]
    std_temperature = data_city_season["season_dev"].iloc[0]

    return abs(current_temperature - mean_temperature) <= 2 * std_temperature

def features_by_city(data: pd.DataFrame) -> pd.DataFrame:
    
    data["rolling_mean"] = data["temperature"].rolling(window=30, min_periods=1).mean()
    data["seasonal_mean"] = data.groupby("season")["temperature"].transform(lambda group: group.mean())
    data["seasonal_std"] = data.groupby("season")["temperature"].transform(lambda group: group.std())
    data["outlier"] = (np.abs(data["temperature"] - data["seasonal_mean"]) > 2 * data["seasonal_std"])

    return data

def paral_features(
    data: pd.DataFrame,
    num_processes: int = 1,
) -> pd.DataFrame:

    cities = data["city"].unique()

    input = [data[data["city"] == city] for city in cities]

    with mp.Pool(processes=num_processes) as pool:
        output = pool.map(features_by_city, input)

    return pd.concat(output, axis=0)

def current_temperature_sync(city: str) -> float:

    params = {
        "appid": API_KEY,
        "q": city,
        "units": "metric"
    }

    url = f"{ENDPOINT_URL}?{urlencode(params)}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    return data["main"]["temp"]


async def current_temperature_async(city: str) -> float:
     
    params = {
        "appid": API_KEY,
        "q": city,
        "units": "metric"
    }

    url = f"{ENDPOINT_URL}?{urlencode(params)}"
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(url) as response:
            data = await response.json()

    return data["main"]["temp"]
