import json
try:
    from .coordinates import *
except:
    from coordinates import *

class FileHandler():
    def __init__(self, file_name: str = 'file.json', encoding: str ='UTF-8') -> None:
        self.file_name = file_name
        self.encoding = encoding

    def write_file(self, coordinates: Coordinates | CartesianCoordinates | SphericalCoordinates) -> None:
        try:
            with open(self.file_name, 'w', encoding='UTF-8') as file:
                json.dump(coordinates.data, file, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"Ошибка записи в файл: {e}")

    def read_file(self) -> None | CartesianCoordinates | SphericalCoordinates:
        try:
            with open(self.file_name, 'r', encoding='UTF-8') as file:
                data = json.load(file)
                if (data['coordinates_type'] == 'cartesian'):
                    return CartesianCoordinates(data['x'], data['y'], data['z'])
                elif (data['coordinates_type'] == 'spherical'):
                    return SphericalCoordinates(data['radius'], data['azimuth_angle'],
                                                data['polar_angle'], data['angle_unit'])
        except (IOError, ValueError) as e:
            print(f"Ошибка чтения файла: {e}")
