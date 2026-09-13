# try:
from geo_transform.viewer import CoordinatesApp, show
# except:
    # from viewer import CoordinatesApp, show

"""
Точка входа для запуска пакета как отдельного приложения.
Использование: python -m geometry
"""

import sys

def main():
    """Главная функция запуска приложения"""
    try:
        show()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()