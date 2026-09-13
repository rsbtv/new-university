from math import sqrt, atan2, sin, cos, pi
from abc import ABC

__all__ = [
    'Coordinates',
    'CartesianCoordinates',
    'SphericalCoordinates',
]
class Coordinates(ABC):
    def transform_angles(self):
        pass
    def to_cartesian(self):
        pass
    def to_spherical(self):
        pass
    def set_x(self, value):
        pass
    def set_y(self, value):
        pass
    def set_z(self, value):
        pass
    def set_radius(self, value):
        pass
    def set_polar_angle(self, value):
        pass
    def set_azimuth_angle(self, value):
        pass
    def __str__(self):
        return "Координаты не заданы"

class CartesianCoordinates(Coordinates):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        """
        Конструктор класса декартовых координат.
        :param x: float = 0; Координата по оси X.
        :param y: float = 0; Координата по оси Y.
        :param z: float = 0; Координата по оси Z.
        """
        self.__x = x
        self.__y = y
        self.__z = z
        self.data = {'coordinates_type': 'cartesian', 'x': self.__x, 'y': self.__y, 'z': self.__z}
    
    def set_x(self, x: float) -> None:
        if self.__x != x:
            self.__x = x
            self.data['x'] = self.__x

    def set_y(self, y: float) -> None:
        if self.__y != y:
            self.__y = y
            self.data['y'] = self.__y

    def set_z(self, z: float) -> None:
        if self.__z != z:
            self.__z = z
            self.data['z'] = self.__z

    def to_spherical(self):
        radius        = sqrt(self.__x ** 2 + self.__y ** 2 + self.__z ** 2)
        azimuth_angle = atan2(self.__x, self.__y + sqrt(self.__x ** 2 + self.__y ** 2)) * 2
        polar_angle   = atan2(sqrt(self.__x ** 2 + self.__y ** 2), self.__z)
        return SphericalCoordinates(radius, azimuth_angle, polar_angle)

    # def update_data(self, data: dict) -> None:
    #     self._data = data
    #     self.set_x(data['x'])
    #     self.set_y(data['y'])
    #     self.set_z(data['z'])

    def __str__(self) -> str:
        return f'Декартовы координаты\n(X: {self.__x}, Y: {self.__y}, Z: {self.__z})'

class SphericalCoordinates(Coordinates):
    def __init__(self, radius: float, azimuth_angle: float, polar_angle: float, angle_unit: str = 'degrees') -> None:
        self.__radius        = radius
        self.__azimuth_angle = azimuth_angle
        self.__polar_angle   = polar_angle
        self.__angle_unit    = angle_unit
        self.data = {'coordinates_type': 'spherical',
                     'radius'          :  self.__radius,
                     'azimuth_angle'   :  self.__azimuth_angle,
                     'polar_angle'     :  self.__polar_angle,
                     'angle_unit'      :  angle_unit}
        self.__update_angle_unit()

    def set_radius(self, radius: float) -> None:
        if self.__radius != radius:
            self.__radius = radius
            self.data['radius'] = self.__radius

    def set_polar_angle(self, polar_angle: float) -> None:
        if self.__polar_angle != polar_angle:
            self.__polar_angle  = polar_angle
            self.data['polar_angle'] = self.__polar_angle

    def set_azimuth_angle(self, azimuth_angle: float) -> None:
        if self.__azimuth_angle != azimuth_angle:
            self.__azimuth_angle  = azimuth_angle
            self.data['azimuth_angle'] = self.__azimuth_angle

    def set_angle_unit(self, angle_unit: str) -> None:
        if self.__angle_unit != angle_unit:
            self.__angle_unit = angle_unit
            self.data['angle_unit'] = self.__angle_unit

    def angle_unit(self) -> str:
        return self.__angle_unit

    def transform_angles(self) -> None:
        if self.__angle_unit == 'degrees':
            self.__azimuth_angle  = 'radians'
            self.__update_angle_unit()
            self.__polar_angle   *= 180 / pi
            self.__azimuth_angle *= 180 / pi
        else:
            self.__azimuth_angle  = 'degrees'
            self.__update_angle_unit()
            self.__polar_angle   *= pi / 180
            self.__azimuth_angle *= pi / 180

    def to_cartesian(self) -> CartesianCoordinates:
        x = self.__radius * sin(self.__polar_angle) * cos(self.__azimuth_angle)
        y = self.__radius * sin(self.__polar_angle) * sin(self.__azimuth_angle)
        z = self.__radius * cos(self.__polar_angle)

        return CartesianCoordinates(x, y, z)

    def __update_angle_unit(self):
        if self.__angle_unit == 'degrees':
            self.__angle_unit_ru = '°'
        else:
            self.__angle_unit_ru = 'рад.'

    def __str__(self) -> str:
        return f'Сферические координаты\n(Радиус: {self.__radius}, полярный угол: {self.__polar_angle}{self.__angle_unit_ru}, азимутальный угол: {self.__azimuth_angle}{self.__angle_unit_ru})'
    # def update_data(self, data: dict) -> None :
    #     self._data           = data
    #     self.set_radius       (data['radius'])
    #     self.set_polar_angle  (data['polar_angle'])
    #     self.set_azimuth_angle(data['azimuth_angle'])
    #     self.set_angle_unit   (data['angle_unit'])

    # def read_file(self):
    #     try:
    #         with open(self._file_name, 'r', encoding='UTF-8') as file:
    #             data = json.load(file)
    #             if (data['coordinates_type'] == 'spherical'):
    #                 self.__update_data(data)
    #             else:
    #                 raise IOError('Неверный формат данных')
    #     except (IOError, ValueError) as e:
    #         print(f"Ошибка чтения файла: {e}")