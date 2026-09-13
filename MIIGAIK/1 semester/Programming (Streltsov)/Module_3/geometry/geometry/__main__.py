if __name__ == "__main__":
    from viewer import GeometryApp
else:
    from .viewer import GeometryApp

"""
Точка входа для запуска пакета как отдельного приложения.
Использование: python -m geometry
"""

import sys

def main():
    """Главная функция запуска приложения"""
    try:
        app = GeometryApp()
        app.show()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()