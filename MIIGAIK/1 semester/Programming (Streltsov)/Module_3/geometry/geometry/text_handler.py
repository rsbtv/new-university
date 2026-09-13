class FileHandler:
    def __init__(self, encoding='UTF-8'):
        self.encoding = encoding

    def write(self, result: str, path: str = 'result.txt'):
        try:
            with open(path, 'a', encoding=self.encoding) as file:
                file.write(result + '\n')
        except IOError as e:
            print(f"Ошибка записи в файл: {e}")

    def read(self, path: str) -> list:
        try:
            with open(path.replace('\\', '/'), 'r', encoding=self.encoding) as file:
                return [[float(num) for num in line.split()]
                       for line in file.readlines()]
        except (IOError, ValueError) as e:
            print(f"Ошибка чтения файла: {e}")
            return []

# Использование
handler = FileHandler()
data = handler.read('input.txt')
handler.write('результат')