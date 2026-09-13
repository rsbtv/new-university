try:
    from .coordinates import *
    from .file_utils import *
except:
    from coordinates import *
    from file_utils import *

# __all__ = ['CoordinatesApp']

class CoordinatesApp:
    def __init__(self):
        self.coordinates = Coordinates()
        self.file_handler = FileHandler()
        # Возможно, стоит соединить эти словари в один
        self.coordinates_menu = {
            1: ('Декартовы',   self.__create_cartesian),
            2: ('Сферические', self.__create_spherical)
        }
        self.start_menu = {
            # 1: ('Создать новый экземпляр', self.show_menu(self.coordinates_menu)),
            1: ('Создать новый экземпляр', self.show_menu, self.coordinates_menu),
            2: ('Загрузить из файла',      self.__load_from_file)
        }
        self.cartesian_attributes_menu = {
            1: ('X', self.__set_attribute, 'X'),
            2: ('Y', self.__set_attribute, 'Y'),
            3: ('Z', self.__set_attribute, 'Z'),
        }
        self.cartesian_menu = {
            1: ('Сохранить в файл',            self.__save_to_file),
            2: ('Перевести в сферические',     self.__cartesian_to_spherical),
            3: ('Изменить один из параметров', self.show_menu, self.cartesian_attributes_menu),
        }
        self.spherical_attributes_menu = {
            1: ('Радиус',                   self.__set_attribute, 'Radius'),
            2: ('Полярный угол',            self.__set_attribute, 'Polar angle'),
            3: ('Азимутальный угол',        self.__set_attribute, 'Azimuth angle'),
            4: ('Перевести значения углов', self.coordinates.transform_angles)
        }
        self.spherical_menu = {
            1: ('Сохранить в файл',            self.__save_to_file),
            2: ('Перевести в декартовы',       self.__spherical_to_cartesian),
            3: ('Изменить один из параметров', self.show_menu, self.spherical_attributes_menu),
        }

    def __transform_angles(self):
        self.coordinates.transform_angles()
        self.show_menu(self.spherical_menu)

    def __get_float(self, prompt: str) -> float:
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Ошибка: введите число")

    def __create_cartesian(self):
        x = self.__get_float("Введите X: ")
        y = self.__get_float("Введите Y: ")
        z = self.__get_float("Введите Z: ")
        self.coordinates = CartesianCoordinates(x, y, z)
        print("Декартовы координаты созданы.")
        self.show_menu(self.cartesian_menu)

    def __create_spherical(self):
        radius        = self.__get_float("Введите радиус: ")
        azimuth_angle = self.__get_float("Введите азимутальный угол: ")
        polar_angle   = self.__get_float("Введите полярный угол: ")
        self.spherical = SphericalCoordinates(radius, azimuth_angle, polar_angle)
        print("Сферические координаты созданы.")
        self.show_menu(self.cartesian_menu)

    def __cartesian_to_spherical(self):
        self.coordinates = self.coordinates.to_spherical()
        self.show_menu(self.spherical_menu)

    def __spherical_to_cartesian(self):
        self.coordinates = self.coordinates.to_cartesian()
        print(self.coordinates)

    def __save_to_file(self):
        self.file_handler.write_file(self.coordinates)
        if type(self.coordinates) == CartesianCoordinates:
            self.show_menu(self.cartesian_menu)
        else:
            self.show_menu(self.spherical_menu)

    def __load_from_file(self):
        self.coordinates = self.file_handler.read_file()
        print(self.coordinates)
        self.show_menu(self.start_menu)

    def show_menu(self, menu: dict):
        print(self.coordinates)
        print("\nВыберите функцию:")
        for key, (value) in menu.items():
            print(f"{key}. {value[0]}")
        try:
            choice = int(input("Введите номер функции: "))
            if choice == 0:
                print("Выход из программы.")
                return
            elif choice in menu:
                func = menu[choice][1]
                if func:
                    try:
                        func(menu[choice][2])
                    except:
                        func()
            else:
                print("Неверный выбор.")
        except ValueError:
            print("Пожалуйста, введите число.")
        except Exception as e:
            print(f"Произошла ошибка: {e}")

    def __set_attribute(self, attribute: str):
        value = self.__get_float(f'Введите {attribute}: ')
        match attribute:
            case 'X':
                self.coordinates.set_x(value)
            case 'Y':
                self.coordinates.set_y(value)
            case 'Z':
                self.coordinates.set_z(value)
            case 'Radius':
                self.coordinates.set_radius(value)
            case 'Polar angle':
                self.coordinates.set_polar_angle(value)
            case 'Azimuth angle':
                self.coordinates.set_azimuth_angle(value)

        if type(self.coordinates) == CartesianCoordinates:
            self.show_menu(self.cartesian_menu)
        else:
            self.show_menu(self.spherical_menu)

def show():
    app = CoordinatesApp()
    app.show_menu(app.start_menu)

if __name__ == "__main__":
    show()