"""
Скрипт нагрузочного тестирования API (формат: multipart/form-data с полем file)
"""

import asyncio
import aiohttp
import time
import numpy as np
from PIL import Image
import io

# === НАСТРОЙКИ ===
API_URL = "http://localhost:8080/predict"
TOTAL_REQUESTS = 1000
CONCURRENT = 50


def generate_test_image_bytes() -> bytes:
    """Генерирует случайное изображение 224x224x3 в PNG и возвращает байты."""
    img = Image.fromarray(
        np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


async def send_request(session: aiohttp.ClientSession,
                       semaphore: asyncio.Semaphore,
                       image_bytes: bytes):
    """Отправляет один запрос и возвращает (успех, латентность)."""
    async with semaphore:
        start = time.perf_counter()
        form = aiohttp.FormData()
        form.add_field(
            "file",
            image_bytes,
            filename="test.png",
            content_type="image/png",
        )
        try:
            async with session.post(
                API_URL,
                data=form,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                # читаем ответ для полноты, но не используем
                await resp.json()
                ok = resp.status == 200
        except Exception:
            ok = False
        latency = time.perf_counter() - start
        return ok, latency


async def main():
    print(f"🚀 Запуск теста: {TOTAL_REQUESTS} запросов, {CONCURRENT} параллельных\n")

    # Один и тот же тестовый образ для всех запросов
    image_bytes = generate_test_image_bytes()

    semaphore = asyncio.Semaphore(CONCURRENT)

    async with aiohttp.ClientSession() as session:
        # Прогрев
        await asyncio.gather(
            *[send_request(session, semaphore, image_bytes) for _ in range(5)]
        )

        # Основной тест
        start_time = time.perf_counter()
        results = await asyncio.gather(
            *[send_request(session, semaphore, image_bytes) for _ in range(TOTAL_REQUESTS)]
        )
        total_time = time.perf_counter() - start_time

    # Подсчёт результатов
    successful_latencies = [r[1] for r in results if r[0]]
    failed = len(results) - len(successful_latencies)

    if not successful_latencies:
        print("❌ Все запросы неуспешны, метрики посчитать нельзя")
        return

    lat_ms = np.array(successful_latencies) * 1000.0
    avg = float(np.mean(lat_ms))
    p50 = float(np.percentile(lat_ms, 50))
    p95 = float(np.percentile(lat_ms, 95))
    p99 = float(np.percentile(lat_ms, 99))
    throughput = TOTAL_REQUESTS / total_time

    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 40)
    print(f"Успешных:    {len(successful_latencies)}/{TOTAL_REQUESTS} "
          f"({len(successful_latencies)/TOTAL_REQUESTS*100:.1f}%)")
    print(f"Неуспешных:  {failed}")
    print(f"Время:       {total_time:.2f} сек")
    print(f"\n⚡ Throughput: {throughput:.2f} RPS")
    print("\n⏱️  Латентность (мс):")
    print(f"   Avg:  {avg:.2f}")
    print(f"   p50:  {p50:.2f}")
    print(f"   p95:  {p95:.2f}")
    print(f"   p99:  {p99:.2f}")


if __name__ == "__main__":
    asyncio.run(main())