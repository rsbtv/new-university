"""
collector.py — модуль сбора геоданных из OSM через osmnx / Overpass API
"""

import os
import time
import logging
import geopandas as gpd
import pandas as pd
import requests

try:
    import osmnx as ox
    from osmnx._errors import InsufficientResponseError
    ox.settings.log_console = False
    ox.settings.use_cache   = True
    # Увеличим таймауты, чтобы Overpass не ронял запросы слишком быстро
    ox.settings.overpass_settings = " [timeout:120][out:json]; "
    ox.settings.timeout = 180
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

from config import (
    CITIES, OSM_BUILDING_TAGS, CRS_GEO, DATA_DIR,
    YEAR_START, YEAR_END
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("collector.log")]
)
log = logging.getLogger(__name__)


def fetch_buildings_osm(city_key: str, city_name: str) -> gpd.GeoDataFrame:
    """
    Загружает все здания для города из OSM.
    При любой сетевой ошибке возвращает пустой GeoDataFrame, чтобы
    основной пайплайн не падал.
    """
    if not OSMNX_AVAILABLE:
        raise ImportError("osmnx не установлен: pip install osmnx")

    log.info(f"[{city_key}] Загрузка зданий из OSM: {city_name}")
    start = time.time()

    try:
        gdf = ox.features_from_place(city_name, tags=OSM_BUILDING_TAGS)
    except (requests.exceptions.RequestException,
            InsufficientResponseError,
            ConnectionError) as e:
        log.error(f"[{city_key}] Ошибка Overpass / соединения: {e}")
        # вернём пустой слой — дальше пайплайн сам пропустит город
        return gpd.GeoDataFrame(geometry=[], crs=CRS_GEO)

    if gdf.empty:
        log.warning(f"[{city_key}] Получен пустой результат от OSM")
        return gdf.to_crs(CRS_GEO)

    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.to_crs(CRS_GEO)

    elapsed = time.time() - start
    log.info(f"[{city_key}] Получено {len(gdf):,} объектов за {elapsed:.1f}с")
    return gdf


def parse_year(value) -> int | None:
    """
    Пытается извлечь год из поля start_date (форматы: '2015', '2015-07', '2015-07-01').
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    try:
        return int(s[:4])
    except (ValueError, IndexError):
        return None


def filter_by_year(gdf: gpd.GeoDataFrame,
                   year_start: int = YEAR_START,
                   year_end:   int = YEAR_END) -> gpd.GeoDataFrame:
    """
    Фильтрует GeoDataFrame по полю start_date, оставляя только объекты
    с известной датой в диапазоне [year_start, year_end].
    """
    if gdf.empty:
        return gdf

    gdf = gdf.copy()
    gdf["year"] = gdf.get("start_date", pd.Series(dtype=object)).apply(parse_year)

    before = len(gdf)
    gdf = gdf[gdf["year"].notna()].copy()
    if gdf.empty:
        log.warning("После фильтрации по year все объекты отфильтрованы")
        return gdf

    gdf["year"] = gdf["year"].astype(int)
    gdf = gdf[(gdf["year"] >= year_start) & (gdf["year"] <= year_end)].copy()
    log.info(f"После фильтрации по году: {len(gdf):,} / {before:,} объектов")
    return gdf


def enrich_geometry(gdf: gpd.GeoDataFrame,
                    crs_metric: str = "EPSG:32637") -> gpd.GeoDataFrame:
    """
    Добавляет производные поля: площадь здания, координаты центроида.
    """
    if gdf.empty:
        return gdf

    gdf_m = gdf.to_crs(crs_metric)
    gdf["area_m2"]      = gdf_m.geometry.area.round(1)
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    return gdf


def save_raw(gdf: gpd.GeoDataFrame, city_key: str) -> str:
    """
    Сохраняет сырые данные в GeoPackage.
    Возвращает путь к файлу.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{city_key}_buildings_raw.gpkg")
    gdf.to_file(path, driver="GPKG")
    log.info(f"Сохранено: {path}")
    return path


def load_or_fetch(city_key: str, force_refresh: bool = False) -> gpd.GeoDataFrame:
    """
    Загружает данные из кэша (GeoPackage) или собирает заново из OSM.
    При ошибке сетевого запроса возвращает пустой GeoDataFrame.
    """
    path = os.path.join(DATA_DIR, f"{city_key}_buildings_raw.gpkg")

    if os.path.exists(path) and not force_refresh:
        log.info(f"[{city_key}] Загрузка из кэша: {path}")
        return gpd.read_file(path)

    city_name = CITIES.get(city_key)
    if not city_name:
        raise ValueError(f"Неизвестный город: {city_key}. Доступны: {list(CITIES.keys())}")

    gdf = fetch_buildings_osm(city_key, city_name)
    if gdf.empty:
        log.warning(f"[{city_key}] Данные не получены, кэш не создаётся")
        return gdf

    gdf = enrich_geometry(gdf)
    save_raw(gdf, city_key)
    return gdf


def collect_all(force_refresh: bool = False) -> dict[str, gpd.GeoDataFrame]:
    """
    Собирает данные для всех городов из config.CITIES.
    Возвращает словарь {city_key: GeoDataFrame}.
    """
    results = {}
    for key in CITIES:
        try:
            results[key] = load_or_fetch(key, force_refresh=force_refresh)
        except Exception as e:
            log.error(f"[{key}] Ошибка сбора: {e}")
    return results