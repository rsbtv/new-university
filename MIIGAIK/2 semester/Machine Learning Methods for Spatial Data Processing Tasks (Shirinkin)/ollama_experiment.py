import ollama
import json
from datetime import datetime
import os

# Конфигурация эксперимента
MODELS = ['gemma3:1b', 'qwen3:4b']
PARAMS_SETS = {
    'strict': {'temperature': 0.1, 'num_ctx': 4096, 'top_p': 0.9, 'repeat_penalty': 1.1},
    'balanced': {'temperature': 0.4, 'num_ctx': 8192, 'top_p': 0.9},
    'creative': {'temperature': 0.9, 'num_ctx': 4096, 'top_p': 0.95}
}

# Промпты для разных экспериментов
PROMPTS = {
    'explain': '''Объясни, как работает бинарный поиск. 
    1) Интуитивное объяснение для новичка (до 5 предложений). 
    2) Формальное описание шагов в виде списка. 
    3) Код на Python в одном блоке.''',

    'bugfix': '''Вот код с ошибкой:
def find_max(lst):
    max_val = 0
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val

1) Что делает код? 2) Найди ошибку. 3) Исправь.''',

    'json': '''Текст: "Петя купил 3 яблока по 10р и 2 апельсина по 15р."
Верни СТРОГО JSON:
{
  "buyer": "string",
  "items": [{"name": "str", "count": int, "price": int}],
  "total": int
}
Только JSON, без текста!''',

    'story': '''Напиши короткий рассказ (200 слов) о программисте, дебажащем LLM ночью. 
    Стиль: нуар, самоирония. Структура: вступление → действие → вывод.'''
}


def run_experiment():
    """Основная функция эксперимента"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ollama_experiment_{timestamp}.md"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Ollama Эксперимент: {timestamp}\n\n")
        f.write("## Конфигурация\n")
        f.write(f"- Модели: {', '.join(MODELS)}\n")
        f.write(f"- Промпты: {len(PROMPTS)} шт.\n")
        f.write(f"- Наборы параметров: {list(PARAMS_SETS.keys())}\n\n")

        results = []

        for model in MODELS:
            f.write(f"## Модель: `{model}`\n\n")

            for param_set_name, params in PARAMS_SETS.items():
                f.write(f"### Режим: {param_set_name} | Параметры: {params}\n\n")

                for prompt_name, prompt_text in PROMPTS.items():
                    print(f"Запуск: {model} | {param_set_name} | {prompt_name}")

                    try:
                        response = ollama.chat(
                            model=model,
                            messages=[{'role': 'user', 'content': prompt_text}],
                            options=params
                        )
                        answer = response['message']['content'].strip()

                        # Сохраняем в файл
                        f.write(f"#### {prompt_name}\n")
                        f.write(f"**Параметры:** `{params}`\n\n")
                        f.write(f"```\n{answer}\n```\n\n")

                        results.append({
                            'model': model,
                            'param_set': param_set_name,
                            'prompt': prompt_name,
                            'answer': answer,
                            'tokens': response.get('eval_count', 0)
                        })

                    except Exception as e:
                        f.write(f"❌ Ошибка: {e}\n\n")
                        print(f"Ошибка: {e}")

                f.write("---\n\n")

        # Итоговая таблица сравнения
        f.write("## Сводная таблица\n\n")
        f.write("| Модель | Режим | Промпт | Токены | Длина ответа |\n")
        f.write("|--------|-------|--------|--------|--------------|\n")

        for r in results:
            f.write(f"| {r['model']} | {r['param_set']} | {r['prompt']} | {r['tokens']} | {len(r['answer'])} |\n")

    print(f"✅ Эксперимент завершён! Результаты в {output_file}")
    return output_file


if __name__ == "__main__":
    # Проверяем, что Ollama запущена
    try:
        ollama.list()
        print("Ollama доступна. Запускаем эксперимент...")
        run_experiment()
    except Exception as e:
        print(f"❌ Ollama не запущена или недоступна: {e}")
        print("Запусти: ollama serve")
