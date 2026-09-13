"""
config.py — настройки системы сбора геоданных строительной активности
"""

# Целевые города / полигоны (можно добавить любые)
CITIES = {
    # "moscow":     "Moscow, Russia",
    "moscow": "Tverskoy District, Moscow, Russia"
    # "spb":        "Saint Petersburg, Russia",
    # "novosibirsk":"Novosibirsk, Russia",
}

# Временной диапазон анализа
YEAR_START = 2010
YEAR_END   = 2024

# OSM-теги для зданий
OSM_BUILDING_TAGS = {"building": True}

# CRS для метрических расчётов (Россия — UTM зоны или пулковская)
CRS_METRIC = "EPSG:32637"   # UTM zone 37N (Москва/ЦФО)
CRS_GEO    = "EPSG:4326"

# KDE
KDE_BANDWIDTH  = 0.15
ANOMALY_ZSCORE = 2.5        # порог z-score для аномалий

# Пути
DATA_DIR   = "data"
OUTPUT_DIR = "output"
REPORT_DIR = "reports"

# Параметры визуализации
MAP_TILES     = "CartoDB dark_matter"
HEATMAP_RADIUS = 12
HEATMAP_BLUR   = 8
