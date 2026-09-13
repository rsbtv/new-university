# GIS Construction Activity Analyzer

Sistema sbora i analiza prostranstvennykh dannykh stroitelnoj
aktivnosti na osnove otkrytykh geodannykh OpenStreetMap.

## Arkhitektura

```
gis_construction_system/
├── config.py        # Nastrojki: goroda, parametry KDE, puti
├── collector.py     # Sbor dannykh iz OSM (osmnx / Overpass API)
├── analyzer.py      # KDE-analiz, z-score, klasterizatsiya anomalij
├── visualizer.py    # Karty Matplotlib + interaktivnye karty Folium
├── reporter.py      # CSV / GeoPackage / tekstovyj otchot
├── main.py          # Tochka vkhoda (CLI)
├── data/            # Keshirovannyye geodannye (GeoPackage)
├── output/          # PNG-karty, HTML-karty, CSV-rezul'taty
└── reports/         # Tekstovye otchoty, statistika po godam
```

## Bystriy start

```bash
# Ustanovka zavisimostej
pip install -r requirements.txt

# Analiz Moskvy
python main.py --city moscow

# Vse goroda iz config.py
python main.py

# Peresobranie dannykh iz OSM (ignorirovat' kesh)
python main.py --city moscow --refresh

# Bez renderinga PNG (bystree)
python main.py --city moscow --no-maps
```

## Vykhodnyye fajly

| Fajl | Opisaniye |
|------|-----------|
| `output/{city}_kde_{year}.png` | Karta KDE-plotnosti za god |
| `output/{city}_dynamics.png` | Graf dinamiki zastrojki |
| `output/{city}_anomaly_map.html` | Interaktivnaya karta anomalij |
| `output/{city}_anomalies.csv` | CSV s koordinatami anomalij |
| `output/{city}_buildings_analyzed.gpkg` | Polnyye geodannye (QGIS) |
| `reports/{city}_yearly_stats.csv` | Statistika po godam |
| `reports/{city}_report.txt` | Tekstovyj summary |

## Dobavleniye goroda

V `config.py`:
```python
CITIES = {
    "moscow":     "Moscow, Russia",
    "spb":        "Saint Petersburg, Russia",
    "your_city":  "Yekaterinburg, Russia",   # <-- dobavit'
}
```

## Metodologiya

1. **Sbor**: `osmnx.features_from_place()` + fil'tratsiya po `start_date`
2. **KDE**: `scipy.stats.gaussian_kde` s bandwidth=0.15 na koordinatakh tsentroidov
3. **Anomalii**: z-score > 2.5 sigma ot srednej plotnosti
4. **Klasterizatsiya**: DBSCAN (sklearn) s eps=0.01 grad (~1 km)
5. **Vizualizatsiya**: Matplotlib (PNG) + Folium/Leaflet (HTML)
