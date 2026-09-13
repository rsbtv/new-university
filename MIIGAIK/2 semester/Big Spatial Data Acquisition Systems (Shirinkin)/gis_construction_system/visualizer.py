"""
visualizer.py — визуализация результатов:
    - Статичные карты KDE (Matplotlib)
    - Интерактивная тепловая карта (Folium)
    - Графики динамики застройки
    - HTML-отчёт
"""

import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from config import (
    OUTPUT_DIR, REPORT_DIR,
    MAP_TILES, HEATMAP_RADIUS, HEATMAP_BLUR,
    YEAR_START, YEAR_END
)
from analyzer import compute_kde_grid

log = logging.getLogger(__name__)


# ── Статичные карты ────────────────────────────────────────────────────────────

def plot_kde_year(gdf: gpd.GeoDataFrame, year: int,
                  city_key: str = "city",
                  save: bool = True) -> str | None:
    """
    Строит карту KDE-плотности для одного года.
    """
    yr_gdf = gdf[gdf["year"] == year]
    if len(yr_gdf) < 5:
        return None

    coords = np.vstack([
        yr_gdf.geometry.centroid.x.values,
        yr_gdf.geometry.centroid.y.values
    ]).T

    xx, yy, density = compute_kde_grid(coords)
    if density is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0a0a0f")
    ax.set_facecolor("#0a0a0f")

    norm = Normalize(vmin=0, vmax=density.max())
    ax.contourf(xx, yy, density, levels=20, cmap="plasma", norm=norm, alpha=0.85)
    ax.scatter(coords[:, 0], coords[:, 1],
               s=1, c="#06b6d4", alpha=0.15, linewidths=0)

    ax.set_title(f"{city_key.upper()} — {year}: KDE plotnost zastrojki",
                 color="white", fontsize=14, pad=10)
    ax.tick_params(colors="gray"); ax.set_xlabel("Lon", color="gray")
    ax.set_ylabel("Lat", color="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e2e")

    plt.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"{city_key}_kde_{year}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor="#0a0a0f")
        plt.close()
        return path
    plt.show()
    return None


def plot_all_years(gdf: gpd.GeoDataFrame, city_key: str = "city"):
    """Строит PNG-карты для каждого года в диапазоне."""
    paths = []
    for year in range(YEAR_START, YEAR_END + 1):
        p = plot_kde_year(gdf, year, city_key)
        if p:
            paths.append(p)
            log.info(f"Сохранена карта: {p}")
    return paths


def plot_dynamics(stats: pd.DataFrame, city_key: str = "city") -> str:
    """
    График динамики строительной активности по годам.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor="#0a0a0f")

    for ax in axes:
        ax.set_facecolor("#12121a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e1e2e")

    # 1. Количество зданий
    axes[0].bar(stats["year"], stats["count"],
                color="#06b6d4", alpha=0.8, width=0.7)
    axes[0].plot(stats["year"], stats["count"],
                 color="white", linewidth=1.5, marker="o", markersize=4)
    axes[0].set_title(f"{city_key.upper()} — kolichestvo novykh zdanij po godam",
                      color="white", fontsize=13)
    axes[0].set_ylabel("Kol-vo zdanij", color="#94a3b8")
    axes[0].grid(axis="y", color="#1e1e2e", linewidth=0.5)

    # 2. Аномалии
    if "anomalies" in stats.columns and stats["anomalies"].notna().any():
        axes[1].bar(stats["year"], stats["anomalies"],
                    color="#7c3aed", alpha=0.85, width=0.7)
        axes[1].set_title("Anomal'nye zony po godam (z-score > 2.5)",
                          color="white", fontsize=13)
        axes[1].set_ylabel("Anomalii", color="#94a3b8")
        axes[1].grid(axis="y", color="#1e1e2e", linewidth=0.5)
    else:
        axes[1].text(0.5, 0.5, "Dannyye anomalij otsutstvuyut",
                     transform=axes[1].transAxes,
                     ha="center", va="center", color="#94a3b8")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{city_key}_dynamics.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close()
    log.info(f"График динамики: {path}")
    return path


# ── Интерактивная карта (Folium) ───────────────────────────────────────────────

def make_interactive_map(gdf: gpd.GeoDataFrame,
                          city_key: str = "city") -> str | None:
    """
    Создаёт интерактивную HTML-карту с тепловой картой аномалий.
    """
    if not FOLIUM_AVAILABLE:
        log.warning("folium не установлен: pip install folium")
        return None

    anom = gdf[gdf.get("is_anomaly", pd.Series(False, index=gdf.index)) == True]
    if len(anom) == 0:
        log.warning("Аномалий нет — карта не будет информативной")
        anom = gdf  # покажем все

    center_lat = gdf.geometry.centroid.y.mean()
    center_lon = gdf.geometry.centroid.x.mean()

    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=11,
                   tiles=MAP_TILES)

    # Тепловая карта аномалий
    heat_data = [
        [row.geometry.centroid.y, row.geometry.centroid.x,
         float(row.get("z_score", 1)) if hasattr(row, "get") else 1.0]
        for _, row in anom.iterrows()
    ]
    HeatMap(heat_data,
            radius=HEATMAP_RADIUS,
            blur=HEATMAP_BLUR,
            min_opacity=0.4).add_to(m)

    # Маркеры кластеров аномалий (топ-50)
    mc = MarkerCluster(name="Anomal'nye zdaniya").add_to(m)
    for _, row in anom.head(50).iterrows():
        lat = row.geometry.centroid.y
        lon = row.geometry.centroid.x
        popup_text = (
            f"<b>God:</b> {row.get('year', '?')}<br>"
            f"<b>Z-score:</b> {row.get('z_score', '?'):.2f}<br>"
            f"<b>Ploshchad:</b> {row.get('area_m2', '?')} m2"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#06b6d4",
            fill=True,
            fill_color="#06b6d4",
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(mc)

    folium.LayerControl().add_to(m)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{city_key}_anomaly_map.html")
    m.save(path)
    log.info(f"Интерактивная карта: {path}")
    return path
