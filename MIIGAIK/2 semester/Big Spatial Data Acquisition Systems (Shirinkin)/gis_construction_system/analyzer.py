"""
analyzer.py — пространственно-временной анализ строительной активности:
    - Kernel Density Estimation (KDE) по годам
    - Обнаружение аномалий через z-score
    - Статистика по временным срезам
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import gaussian_kde
from shapely.geometry import Point

from config import (
    KDE_BANDWIDTH, ANOMALY_ZSCORE,
    YEAR_START, YEAR_END, CRS_GEO
)

log = logging.getLogger(__name__)


def compute_kde_grid(points_xy: np.ndarray,
                     grid_size: int = 200,
                     bandwidth: float = KDE_BANDWIDTH):
    """
    Рассчитывает KDE на регулярной сетке.

    Args:
        points_xy: массив [[lon, lat], ...] формы (N, 2)
        grid_size:  разрешение сетки (grid_size x grid_size)
        bandwidth:  параметр сглаживания KDE

    Returns:
        xx, yy, density — сетки и значения плотности
    """
    if len(points_xy) < 5:
        return None, None, None

    x, y = points_xy[:, 0], points_xy[:, 1]
    kde  = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)

    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    xx, yy = np.meshgrid(xi, yi)

    positions = np.vstack([xx.ravel(), yy.ravel()])
    density   = kde(positions).reshape(grid_size, grid_size)
    return xx, yy, density


def detect_point_anomalies(gdf: gpd.GeoDataFrame,
                            bandwidth: float = KDE_BANDWIDTH,
                            zscore_threshold: float = ANOMALY_ZSCORE) -> gpd.GeoDataFrame:
    """
    Для каждого здания считает локальную KDE-плотность и z-score.
    Здания с z-score > порога помечаются как аномалии.

    Returns:
        GeoDataFrame с добавленными полями: kde_density, z_score, is_anomaly
    """
    if len(gdf) < 10:
        log.warning("Слишком мало точек для KDE-анализа")
        gdf["kde_density"] = np.nan
        gdf["z_score"]     = np.nan
        gdf["is_anomaly"]  = False
        return gdf

    coords = np.vstack([
        gdf.geometry.centroid.x.values,
        gdf.geometry.centroid.y.values
    ])
    kde     = gaussian_kde(coords, bw_method=bandwidth)
    density = kde(coords)

    mean = density.mean()
    std  = density.std()
    z    = (density - mean) / std if std > 0 else np.zeros_like(density)

    gdf = gdf.copy()
    gdf["kde_density"] = density.round(8)
    gdf["z_score"]     = z.round(4)
    gdf["is_anomaly"]  = z > zscore_threshold

    n_anom = gdf["is_anomaly"].sum()
    log.info(f"Обнаружено аномалий: {n_anom:,} / {len(gdf):,} ({100*n_anom/len(gdf):.1f}%)")
    return gdf


def yearly_stats(gdf: gpd.GeoDataFrame,
                 year_start: int = YEAR_START,
                 year_end:   int = YEAR_END) -> pd.DataFrame:
    """
    Статистика по годам: кол-во зданий, средняя площадь, кол-во аномалий.
    """
    rows = []
    for year in range(year_start, year_end + 1):
        yr = gdf[gdf["year"] == year]
        if len(yr) == 0:
            continue
        rows.append({
            "year":        year,
            "count":       len(yr),
            "total_area":  yr["area_m2"].sum() if "area_m2" in yr.columns else None,
            "mean_area":   yr["area_m2"].mean().round(1) if "area_m2" in yr.columns else None,
            "anomalies":   int(yr["is_anomaly"].sum()) if "is_anomaly" in yr.columns else None,
        })
    df = pd.DataFrame(rows)
    df["pct_change"] = df["count"].pct_change().mul(100).round(1)
    return df


def anomaly_hotspots(gdf: gpd.GeoDataFrame,
                     cluster_eps_deg: float = 0.01) -> gpd.GeoDataFrame:
    """
    Упрощённая кластеризация аномалий методом сетки (binning).
    Возвращает GeoDataFrame точек-центроидов кластеров с атрибутами.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        log.warning("sklearn не установлен, кластеризация пропущена")
        return gdf[gdf.get("is_anomaly", False)]

    anom = gdf[gdf["is_anomaly"] == True].copy()
    if len(anom) < 2:
        return anom

    coords = np.radians(np.vstack([
        anom.geometry.centroid.y.values,
        anom.geometry.centroid.x.values
    ]).T)

    eps_rad = cluster_eps_deg * np.pi / 180
    labels  = DBSCAN(eps=eps_rad, min_samples=3,
                     algorithm="ball_tree",
                     metric="haversine").fit_predict(coords)

    anom["cluster_id"] = labels
    log.info(f"Кластеров аномалий: {len(set(labels)) - (1 if -1 in labels else 0)}")
    return anom


def run_full_analysis(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Полный цикл анализа:
        1. Обнаружение аномалий (KDE + z-score)
        2. Кластеризация горячих точек
        3. Годовая статистика

    Returns:
        (enriched_gdf, stats_df)
    """
    log.info("Запуск полного анализа...")
    gdf_analyzed = detect_point_anomalies(gdf)
    gdf_analyzed = anomaly_hotspots(gdf_analyzed)
    stats        = yearly_stats(gdf_analyzed)
    log.info("Анализ завершён")
    return gdf_analyzed, stats
