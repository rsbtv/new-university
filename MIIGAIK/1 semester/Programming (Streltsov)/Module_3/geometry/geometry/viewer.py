try:
    from .figure import Square, Rectangle, Triangle, Polygon
    from .text_handler import FileHandler
except:
    from figure import Square, Rectangle, Triangle, Polygon
    from text_handler import FileHandler

__all__ = ['GeometryApp']

class GeometryApp:
    def __init__(self):
        self.file_handler = FileHandler()
        self.menu_options = {
            1: ("Площадь квадрата", self._square_area),
            2: ("Площадь прямоугольника", self._rectangle_area),
            3: ("Площадь треугольника", self._triangle_area),
            4: ("Периметр произвольной геометрической фигуры", self._polygon_perimeter),
            5: ("Периметр фигур из файла", self._file_perimeter),
            6: ("Выход", None)
        }

    def _get_float(self, prompt: str) -> float:
        """Безопасный ввод числа"""
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Ошибка: введите число")

    def _process_result(self, result: str):
        """Вывод и запись результата"""
        print(result)
        self.file_handler.write(result)

    def _square_area(self):
        side = self._get_float('Введите длину стороны квадрата: ')
        area = Square(side).area()
        self._process_result(f"Площадь квадрата: {area}")

    def _rectangle_area(self):
        width = self._get_float('Введите ширину прямоугольника: ')
        height = self._get_float('Введите высоту прямоугольника: ')
        area = Rectangle(width, height).area()
        self._process_result(f"Площадь прямоугольника: {area}")

    def _triangle_area(self):
        try:
            a = self._get_float('Введите длину стороны a: ')
            b = self._get_float('Введите длину стороны b: ')
            c = self._get_float('Введите длину стороны c: ')
            area = Triangle(a, b, c).area()
            self._process_result(f"Площадь треугольника: {area}")
        except ValueError as e:
            print(f"Ошибка создания треугольника: {e}")

    def _polygon_perimeter(self):
        lengths = input('Введите длины сторон через пробел: ').split()
        try:
            lengths = [float(l) for l in lengths]
            perimeter = Polygon(lengths).perimetr()
            self._process_result(f"Периметр фигуры: {perimeter}")
        except ValueError as e:
            print(f"Ошибка: {e}")

    def _file_perimeter(self):
        path = input('Введите путь к файлу: ')
        data = self.file_handler.read(path)
        for i, lengths in enumerate(data, 1):
            try:
                perimeter = Polygon(lengths).perimetr()
                self._process_result(f"Периметр {i}-й фигуры: {perimeter}")
            except ValueError as e:
                print(f"Ошибка в фигуре {i}: {e}")

    def show(self):
        """Главный цикл приложения"""
        while True:
            print("\nВыберите функцию:")
            for key, (desc, _) in self.menu_options.items():
                print(f"{key}. {desc}")

            try:
                choice = int(input("Введите номер программы (1-6): "))
                if choice == 6:
                    print("Выход из программы.")
                    break
                elif choice in self.menu_options:
                    self.menu_options[choice][1]()
                else:
                    print("Неверный выбор. Выберите число от 1 до 6.")
            except ValueError:
                print("Пожалуйста, введите число.")
            except Exception as e:
                print(f"Произошла ошибка: {e}")

# Для совместимости со старым кодом
def show():
    app = GeometryApp()
    app.show()