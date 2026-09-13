from .figure import Figure, Polygon, Square, Rectangle, Triangle
from .text_handler import FileHandler
from .viewer import GeometryApp, show

__all__ = [
    # Геометрические фигуры
    'Figure',
    'Polygon',
    'Square',
    'Rectangle',
    'Triangle',
    # Обработчик файлов
    'FileHandler',
    # Приложение
    'GeometryApp',
]