"""
main.py — точка входа в систему сбора и анализа пространственных данных

Использование:
    python main.py                          # анализ всех городов
    python main.py --city moscow            # только Москва
    python main.py --city moscow --refresh  # сбор заново, игнорируя кэш
    python main.py --city moscow --no-maps  # без рендеринга карт
"""

import argparse
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GIS: Analiz stroitelnoj aktivnosti na osnove OSM"
    )
    parser.add_argument("--city", default=None,
                        help="Klyuch goroda iz config.CITIES (napr. moscow)")
    parser.add_argument("--refresh", action="store_true",
                        help="Peresobraniyu dannyye iz OSM (ignorirovat' kesh)")
    parser.add_argument("--no-maps", action="store_true",
                        help="Ne renderit' PNG-karty (bystree)")
    return parser.parse_args()


def run_pipeline(city_key: str,
                 force_refresh: bool = False,
                 render_maps: bool = True):
    """
    Запускает полный pipeline для одного города.
    """
    from collector  import load_or_fetch, filter_by_year
    from analyzer   import run_full_analysis
    from visualizer import plot_all_years, plot_dynamics, make_interactive_map
    from reporter   import save_results, generate_text_report

    log.info(f"{'='*60}")
    log.info(f"GOROD: {city_key.upper()}")
    log.info(f"{'='*60}")

    # 1. Сбор данных
    gdf_raw      = load_or_fetch(city_key, force_refresh=force_refresh)
    gdf_filtered = filter_by_year(gdf_raw)

    if len(gdf_filtered) == 0:
        log.warning(f"[{city_key}] Nyet dannykh posle fil'tratsii — propusk")
        return

    # 2. Анализ
    gdf_analyzed, stats = run_full_analysis(gdf_filtered)

    # 3. Визуализация
    map_paths = []

    if render_maps:
        png_paths = plot_all_years(gdf_analyzed, city_key)
        map_paths.extend(png_paths)
        dyn_path  = plot_dynamics(stats, city_key)
        map_paths.append(dyn_path)

    html_path = make_interactive_map(gdf_analyzed, city_key)
    if html_path:
        map_paths.append(html_path)

    # 4. Сохранение и отчёт
    save_results(gdf_analyzed, stats, city_key)
    generate_text_report(gdf_analyzed, stats, city_key, map_paths)

    log.info(f"[{city_key}] Pipeline zavershyon. Fajlov: {len(map_paths)}")


def main():
    args = parse_args()

    from config import CITIES
    cities_to_run = [args.city] if args.city else list(CITIES.keys())

    for city_key in cities_to_run:
        try:
            run_pipeline(
                city_key      = city_key,
                force_refresh = args.refresh,
                render_maps   = not args.no_maps,
            )
        except Exception as e:
            log.error(f"[{city_key}] Oshibka v pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()
