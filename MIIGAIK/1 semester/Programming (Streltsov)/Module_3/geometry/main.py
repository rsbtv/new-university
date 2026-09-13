# main.py

# Импортируем классы из модуля figure
from geometry.figure import Square, Rectangle, Triangle

# Импортируем класс FileHandler из модуля text_handler
from geometry.text_handler import FileHandler

# Импортируем класс GeometryApp и функцию show из модуля viewer
from geometry.viewer import GeometryApp, show

# Создаем экземпляры классов и используем их методы
square = Square(5)
print(f"Периметр квадрата со стороной 5: {square.perimetr()}")
print(f"Площадь квадрата со стороной 5: {square.area()}")

rectangle = Rectangle(4, 6)
print(f"Периметр прямоугольника со сторонами 4 и 6: {rectangle.perimetr()}")
print(f"Площадь прямоугольника со сторонами 4 и 6: {rectangle.area()}")

triangle = Triangle(3, 4, 5)
print(f"Периметр треугольника со сторонами 3, 4, 5: {triangle.perimetr()}")
print(f"Площадь треугольника со сторонами 3, 4, 5: {triangle.area()}")

# Используем класс FileHandler из модуля text_handler
file_handler = FileHandler()

# Записываем результат в файл
result = "Тестовая строка для записи в файл"
file_handler.write(result)

# Читаем данные из файла
read_data = file_handler.read('input.txt')
print(f"Прочитанные данные из файла: {read_data}")

# Использование функции show из модуля viewer (раскомментируйте следующую строку, чтобы запустить интерфейс)
# show()

# Или можно создать экземпляр GeometryApp напрямую
# app = GeometryApp()
# app.show()