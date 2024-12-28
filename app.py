import streamlit as st
import time
import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# Маппинг месяца на сезон
month_to_season = {
    1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn",
    11: "autumn", 12: "winter",
}

# Основная функция приложения
def main():
    st.title("Анализ погоды")

    # Загрузка данных
    st.header("1. Загрузите данные")
    uploaded_file = st.file_uploader("Загрузите датасет (csv)", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        st.write(data.head())

        # Выбор города
        st.header("2. Выберите город для просмотра")
        cities = data["city"].unique().tolist()
        city = st.selectbox("Выберите город", cities)

        if city:
            st.write(f"Выбранный город: {city}")
            data_city = data[data["city"] == city]
            data_city = add_features(data_city)

            # Статистики
            st.subheader("Описательная статистика")
            st.write(data_city["temperature"].describe())

            st.subheader("Сезонные статистики")
            seasonal_stats = calculate_seasonal_statistics(data_city)
            st.write(seasonal_stats)

            # График исторических температур
            st.subheader("График исторических температур")
            plot_temperature_by_season(data_city)

            # Запрос текущей температуры
            st.header("3. Получение текущей температуры")
            api_key = st.text_input("Введите OpenWeatherAPI key", type="password")

            if api_key:
                st.write("API ключ загружен.")
                if st.button("Получить данные"):
                    try:
                        current_temperature = get_current_temperature(api_key, city)
                        normal = check_temperature(data_city, current_temperature)

                        with st.spinner("Получение данных... Это может занять несколько секунд"):
                        # Ждём 5 секунд
                            time.sleep(5)
                        st.metric(f"Текущая температура в {city}", f"{current_temperature}°C")
                        st.metric("Температура аномальна?", "Нет" if normal else "Да")

                    except HTTPError as e:
                        handle_api_error(e)

# Функции обработки данных
def add_features(data_city: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет новые признаки в данные: скользящее среднее, сезонные статистики и аномалии.
    """
    # Добавление сезонных статистик
    data_city["seasonal_mean"] = data_city.groupby("season")["temperature"].transform("mean")
    data_city["seasonal_std"] = data_city.groupby("season")["temperature"].transform("std")

    # Выявление выбросов
    data_city["outlier"] = np.abs(data_city["temperature"] - data_city["seasonal_mean"]) > 2 * data_city["seasonal_std"]
    return data_city

def calculate_seasonal_statistics(data_city: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает сезонные статистики для каждого сезона в городе.
    """
    seasons = ["winter", "spring", "summer", "autumn"]
    data_seasonal = []

    for season in seasons:
        seasonal_data = data_city[data_city["season"] == season]
        mean = seasonal_data["seasonal_mean"].mean()
        std = seasonal_data["seasonal_std"].mean()
        data_seasonal.append({"season": season, "seasonal_mean": mean, "seasonal_std": std})

    return pd.DataFrame(data_seasonal)

def plot_temperature_by_season(data_city: pd.DataFrame):
    """
    Строит график динамики температуры по сезонам для выбранного города.
    Для каждого сезона будет отображён график изменения температуры по времени.
    """
    # Создаём фигуру и несколько осей для каждого сезона
    seasons = data_city["season"].unique()
    num_seasons = len(seasons)
    
    fig, axes = plt.subplots(num_seasons, 1, figsize=(10, 5 * num_seasons))
    
    if num_seasons == 1:
        axes = [axes]  # Чтобы не было ошибок, если только один сезон

    for i, season in enumerate(seasons):
        ax = axes[i]
        season_data = data_city[data_city["season"] == season]
        
        # Строим график изменения температуры по времени для этого сезона
        sns.lineplot(data=season_data, x="timestamp", y="temperature", ax=ax, marker="o", color="b", label=season)
        
        ax.set_title(f"Temperature Dynamics in {season.capitalize()}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()

    # Автоматически подстраиваем layout
    plt.tight_layout()
    st.pyplot(fig)

def get_current_temperature(api_key: str, city: str) -> float:
    """
    Получает текущую температуру для указанного города.
    """
    endpoint_url = "https://api.openweathermap.org/data/2.5/weather"
    url = f"{endpoint_url}?appid={api_key}&q={city}&units=metric"

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    return data["main"]["temp"]

def check_temperature(data_city: pd.DataFrame, current_temperature: float) -> bool:
    """
    Проверяет, является ли текущая температура нормальной для выбранного города.
    """
    current_month = datetime.now().month
    current_season = month_to_season[current_month]

    seasonal_data = data_city[data_city["season"] == current_season].iloc[0]
    mean_temperature = seasonal_data["seasonal_mean"]
    std_temperature = seasonal_data["seasonal_std"]

    return abs(current_temperature - mean_temperature) <= 2 * std_temperature

def handle_api_error(e: HTTPError):
    """
    Обрабатывает ошибки при запросах к API.
    """
    if e.response.status_code == 401:
        st.error("Invalid API key. Please check your key.")
    else:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()