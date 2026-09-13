#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Engineering Comparison — Ollama
Задача: классификация отзыва клиента
"""

import ollama
import time
import json

MODEL = "gemma3:1b"
REVIEW = "Доставили быстро, но упаковка была помята, и одна деталь оказалась сломана."
SEPARATOR = "=" * 70

results = {}

def run_prompt(technique_name, messages):
    print(f"\n{SEPARATOR}")
    print(f"  ТЕХНИКА: {technique_name}")
    print(SEPARATOR)
    start = time.time()
    response = ollama.chat(model=MODEL, messages=messages)
    elapsed = round(time.time() - start, 2)
    content = response["message"]["content"]
    print(content)
    print(f"\n[Время ответа: {elapsed} сек.]")
    results[technique_name] = {"response": content, "time_sec": elapsed}
    return content


# ──────────────────────────────────────────────
# 1. ZERO-SHOT
# ──────────────────────────────────────────────
run_prompt("1. Zero-Shot", [
    {
        "role": "user",
        "content": (
            f"Классифицируй отзыв как ПОЗИТИВНЫЙ, НЕГАТИВНЫЙ или НЕЙТРАЛЬНЫЙ и объясни почему.\n\n"
            f"Отзыв: «{REVIEW}»"
        )
    }
])

# ──────────────────────────────────────────────
# 2. FEW-SHOT
# ──────────────────────────────────────────────
run_prompt("2. Few-Shot", [
    {
        "role": "user",
        "content": (
            "Классифицируй отзывы. Примеры:\n\n"
            "Отзыв: «Всё отлично, очень доволен покупкой!»\n"
            "Метка: ПОЗИТИВНЫЙ — клиент выражает явное удовлетворение без негатива.\n\n"
            "Отзыв: «Товар сломан, деньги вернуть невозможно, ужасный сервис.»\n"
            "Метка: НЕГАТИВНЫЙ — явный негатив по нескольким аспектам.\n\n"
            "Отзыв: «Получил посылку вовремя.»\n"
            "Метка: НЕЙТРАЛЬНЫЙ — констатация факта без выраженной оценки.\n\n"
            f"Теперь классифицируй:\n"
            f"Отзыв: «{REVIEW}»\n"
            "Метка:"
        )
    }
])

# ──────────────────────────────────────────────
# 3. CHAIN-OF-THOUGHT
# ──────────────────────────────────────────────
run_prompt("3. Chain-of-Thought", [
    {
        "role": "user",
        "content": (
            "Классифицируй отзыв как ПОЗИТИВНЫЙ, НЕГАТИВНЫЙ или НЕЙТРАЛЬНЫЙ.\n"
            "Сначала пошагово разбери каждый аспект отзыва, затем вынеси итоговую метку.\n\n"
            f"Отзыв: «{REVIEW}»\n\n"
            "Рассуждение:"
        )
    }
])

# ──────────────────────────────────────────────
# 4. ROLE PROMPTING
# ──────────────────────────────────────────────
run_prompt("4. Role Prompting", [
    {
        "role": "system",
        "content": (
            "Ты — старший аналитик службы контроля качества маркетплейса с 10-летним опытом. "
            "Ты всегда структурируешь ответ строго по пунктам: "
            "1) Метка, 2) Позитивные аспекты, 3) Негативные аспекты, 4) Рекомендация продавцу."
        )
    },
    {
        "role": "user",
        "content": f"Классифицируй и проанализируй отзыв: «{REVIEW}»"
    }
])

# ──────────────────────────────────────────────
# 5. SELF-REFINE
# ──────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print(f"  ТЕХНИКА: 5. Self-Refine")
print(SEPARATOR)
start = time.time()

messages_sr = [
    {"role": "user", "content": f"Классифицируй отзыв: «{REVIEW}»"}
]
r1 = ollama.chat(model=MODEL, messages=messages_sr)
first_answer = r1["message"]["content"]
print("[Первичный ответ]")
print(first_answer)

messages_sr.append({"role": "assistant", "content": first_answer})
messages_sr.append({
    "role": "user",
    "content": (
        "Критически оцени свой ответ: что можно улучшить или уточнить? "
        "Дай финальную, исправленную и более точную версию."
    )
})
r2 = ollama.chat(model=MODEL, messages=messages_sr)
final_answer = r2["message"]["content"]
print("\n[Улучшенный ответ]")
print(final_answer)

elapsed = round(time.time() - start, 2)
print(f"\n[Время ответа: {elapsed} сек.]")
results["5. Self-Refine"] = {
    "response": f"[Первичный]\n{first_answer}\n\n[Улучшенный]\n{final_answer}",
    "time_sec": elapsed
}

# ──────────────────────────────────────────────
# 6. TEMPLATE PATTERN
# ──────────────────────────────────────────────
run_prompt("6. Template Pattern", [
    {
        "role": "user",
        "content": (
            f"Проанализируй отзыв и заполни шаблон строго в указанном формате:\n\n"
            f"Отзыв: «{REVIEW}»\n\n"
            "Шаблон:\n"
            "- Метка: [ПОЗИТИВНЫЙ / НЕГАТИВНЫЙ / НЕЙТРАЛЬНЫЙ / СМЕШАННЫЙ]\n"
            "- Позитивные аспекты: [перечисли через запятую]\n"
            "- Негативные аспекты: [перечисли через запятую]\n"
            "- Уверенность: [Низкая / Средняя / Высокая]\n"
            "- Рекомендация для продавца: [1 предложение]"
        )
    }
])

# ──────────────────────────────────────────────
# ИТОГ
# ──────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("  ИТОГОВОЕ ВРЕМЯ ПО ТЕХНИКАМ")
print(SEPARATOR)
for name, data in results.items():
    print(f"  {name}: {data['time_sec']} сек.")

print(f"\n{SEPARATOR}")
print("  ВСЕ РЕЗУЛЬТАТЫ СОХРАНЕНЫ В: results.json")
print(SEPARATOR)

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
