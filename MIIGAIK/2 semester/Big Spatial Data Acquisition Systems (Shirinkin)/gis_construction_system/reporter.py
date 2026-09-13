"""
reporter.py — формирование отчётов:
    - CSV с результатами анализа
    - GeoPackage с обогащёнными данными
    - Текстовый summary-отчёт
"""

import os
import logging
from datetime import datetime
import pandas as pd
import geopandas as gpd

from config import DATA_DIR, OUTPUT_DIR, REPORT_DIR

log = logging.getLogger(__name__)


def save_results(gdf: gpd.GeoDataFrame,
                 stats: pd.DataFrame,
                 city_key: str = "city"):
    """
    Сохраняет все результаты анализа:
        - CSV с аномалиями
        - GeoPackage с полными данными
        - CSV со статистикой по годам
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. Аномалии → CSV
    anom_cols = ["year", "area_m2", "centroid_lon", "centroid_lat",
                 "kde_density", "z_score", "is_anomaly", "cluster_id"]
    anom_cols = [c for c in anom_cols if c in gdf.columns]
    anom_path = os.path.join(OUTPUT_DIR, f"{city_key}_anomalies.csv")
    gdf[anom_cols].to_csv(anom_path, index=False, encoding="utf-8")
    log.info(f"CSV аномалий: {anom_path}")

    # 2. Полные данные → GeoPackage
    gpkg_path = os.path.join(OUTPUT_DIR, f"{city_key}_buildings_analyzed.gpkg")
    # Убираем несериализуемые типы
    cols_to_drop = [c for c in gdf.columns
                    if gdf[c].dtype == object and c not in ["geometry"]]
    gdf_save = gdf.drop(columns=cols_to_drop, errors="ignore")
    gdf_save.to_file(gpkg_path, driver="GPKG")
    log.info(f"GeoPackage: {gpkg_path}")

    # 3. Статистика → CSV
    stats_path = os.path.join(REPORT_DIR, f"{city_key}_yearly_stats.csv")
    stats.to_csv(stats_path, index=False, encoding="utf-8")
    log.info(f"Статистика: {stats_path}")

    return anom_path, gpkg_path, stats_path


def generate_text_report(gdf: gpd.GeoDataFrame,
                          stats: pd.DataFrame,
                          city_key: str = "city",
                          map_paths: list = None) -> str:
    """
    Генерирует текстовый summary-отчёт.
    """
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    anom = gdf[gdf.get("is_anomaly", pd.Series(False, index=gdf.index)) == True]

    peak_year  = stats.loc[stats["count"].idxmax(), "year"] if len(stats) else "N/A"
    peak_count = stats["count"].max() if len(stats) else 0

    report = f"""
================================================================================
  ОТЧЁТ: АНАЛИЗ СТРОИТЕЛЬНОЙ АКТИВНОСТИ — {city_key.upper()}
  Сгенерирован: {ts}
================================================================================

1. ОБЩАЯ СТАТИСТИКА
   Всего объектов (с датой):   {len(gdf):>10,}
   Аномальных зон:           {len(anom):>10,}
   Процент аномалий:         {100*len(anom)/max(len(gdf),1):>9.1f}%
   Период анализа:            {stats["year"].min() if len(stats) else "?"} — {stats["year"].max() if len(stats) else "?"}

2. ПИКИ АКТИВНОСТИ
   Макимум зданий за год:    {peak_year} ({peak_count:,} ob.)
"""

    if "pct_change" in stats.columns:
        max_growth = stats.loc[stats["pct_change"].idxmax()]
        report += (
            f"   Наибольший рост:         "
            f"{int(max_growth['year'])} (+{max_growth['pct_change']:.1f}%)"
        )

    report += """
3. КЛЮЧЕВЫЕ ЗОНЫ АНОМАЛИЙ (топ-10 по z-score)
"""
    top10 = (gdf[gdf["is_anomaly"] == True]
             .nlargest(10, "z_score")[["year", "centroid_lat",
                                       "centroid_lon", "z_score", "area_m2"]]
             if "z_score" in gdf.columns and len(anom) > 0
             else pd.DataFrame())

    if len(top10):
        report += top10.to_string(index=False) + ""
    else:
        report += "   Аномалий не обнаружено"

    if map_paths:
        report += "4. СФОРМИРОВАННЫЕ ФАЙЛЫ"
        for p in map_paths:
            report += f"   {p}"

    report += """
================================================================================
  Методология: osmnx + GeoPandas + Gaussian KDE + z-score (порог 2.5 sigma)
================================================================================
"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{city_key}_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Отчёт: {path}")
    print(report)
    return path
