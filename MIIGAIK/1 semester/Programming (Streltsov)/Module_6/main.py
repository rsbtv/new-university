import os
from pdf2docx import parse
from docx2pdf import convert
from PIL import Image
import argparse


def pdf2docx(arg, dir=None):
    if arg == 'all':
        if dir and os.path.isdir(dir):
            for file_name in os.listdir(dir):
                if file_name.endswith('.pdf'):
                    input_path = os.path.join(dir, file_name)
                    new_path = os.path.join(dir, file_name.replace('.pdf', '.docx'))
                    parse(input_path, new_path)
        else:
            print('Директории не существует!')
    else:
        file_path = arg
        try:
            if file_path.endswith('.pdf'):
                new_path = file_path.replace('.pdf', '.docx')
                parse(file_path, new_path)
            else:
                print('Это не pdf-файл!')
        except Exception as e:
            print(e)

def docx2pdf(arg, dir=None):
    if arg == 'all':
        if dir and os.path.isdir(dir):
            for file_name in os.listdir(dir):
                if file_name.endswith('.docx'):
                    input_path = os.path.join(dir, file_name)
                    new_path = os.path.join(dir, file_name.replace('.docx', '.pdf'))
                    convert(input_path, new_path)
        else:
            print('Директории не существует!')
    else:
        file_path = arg
        try:
            if file_path.endswith('.docx'):
                new_path = file_path.replace('.docx', '.pdf')
                convert(file_path, new_path.replace('.docx', '.pdf'))
            else:
                print('Это не docx-файл!')
        except Exception as e:
            print(e)


def compress_images(arg, crop=75, dir=None):
    if arg == 'all':
        if dir and os.path.isdir(dir):
            for file_path in os.listdir(dir):
                if file_path.endswith(('.jpg', '.jpeg')):
                    image_path = os.path.join(dir, file_path)

                    ext = os.path.splitext(image_path)[1].lower()
                    base = os.path.splitext(image_path)[0]
                    compressed_path = f"{base}_compressed{ext}"

                    with Image.open(image_path) as img:
                        img.save(compressed_path, quality=100 - crop)
                elif file_path.endswith('.png'):
                    level = 9 - int(crop / 100 * 9)
                    level = max(0, min(level, 9))

                    image_path = os.path.join(dir, file_path)

                    ext = os.path.splitext(image_path)[1].lower()
                    base = os.path.splitext(image_path)[0]
                    compressed_path = f"{base}_compressed{ext}"

                    with Image.open(image_path) as img:
                        img.save(compressed_path, optimize=True, compress_level=int(level))
        else:
            print('Директории не существует!')
    else:
        image_path = arg

        ext = os.path.splitext(image_path)[1]
        base = os.path.splitext(image_path)[0]
        compressed_path = f"{base}_compressed{ext}"
        print(f'ext: {ext}')
        print(f'base: {base}')
        print(f'compressed: {compressed_path}')
        print(f'image: {image_path}')

        try:
            if (image_path.endswith(('.jpg', '.jpeg'))):
                with Image.open(image_path) as img:
                    img.save(compressed_path, quality=100 - crop)
            elif image_path.endswith('.png'):
                level = int(crop / 100 * 9)

                with Image.open(image_path) as img:
                    img.save(compressed_path, optimize=True, compress_level=level)
            else:
                print('Это не изображение!')
        except Exception as e:
            print(e)


def delete_files_console(mode, mask, dir):
    if not (mode and mask and dir):
        print('Параметры для удаления файлов: --mode, --pattern и --dir')
        return

    if not os.path.isdir(dir):
        print('Директории не существует')
        return

    files = os.listdir(dir)
    deleted = False

    for file_name in files:
        full_path = os.path.join(dir, file_name)
        if not os.path.isfile(full_path):
            continue

        match mode:
            case 'startswith':
                condition = file_name.startswith(mask)
            case 'endswith':
                condition = file_name.endswith(mask)
            case 'contains':
                condition = mask in file_name
            case 'extension':
                condition = os.path.splitext(file_name)[1] == mask
            case _:
                print('Неверный режим удаления')
                return

        if condition:
            os.remove(full_path)
            print(f'Удален файл {full_path}')
            deleted = True

    if not deleted:
        print('Файлы для удаления не найдены!')


def main():
    parser = argparse.ArgumentParser(description='Утилита для обработки файлов')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pdf2docx', metavar='FILE_OR_ALL',
                       help='Конвертировать pdf в docx (для папки - all)')
    group.add_argument('--docx2pdf', metavar='FILE_OR_ALL',
                       help='Конвертировать docx в pdf (для папки - all)')
    group.add_argument('--compress-images', metavar='FILE_OR_ALL',
                       help='Сжать изображения (для папки - all)')
    group.add_argument('--delete', action='store_true', help='Удалить файлы по маске')
    group.add_argument('-i', '--interface', action='store_true', help='Вызвать меню (по умолчанию)')

    parser.add_argument('--crop', type=int, default=75,
                        help='Степень сжатия изображений (по умолчанию 75, только для --compress-images)')
    parser.add_argument('--dir',
                        help='Рабочая папка')
    parser.add_argument('--mode',
                        choices=['startswith', 'endswith', 'contains', 'extension'],
                        help='Маска для удаления')
    parser.add_argument('--pattern',
                        help='Подстрока или расширение для удаления')

    args = parser.parse_args()

    # --interactive -> interface
    if not any([args.pdf2docx, args.docx2pdf, args.compress_images, args.delete, args.interface]):
        args.interface = True

    # Важно! args.dir, а не args.workdir!
    if args.pdf2docx:
        pdf2docx(args.pdf2docx, args.dir)
    elif args.docx2pdf:
        docx2pdf(args.docx2pdf, args.dir)
    elif args.compress_images:
        compress_images(args.compress_images, crop=args.crop, dir=args.dir)
    elif args.delete:
        delete_files_console(args.mode, args.pattern, args.dir)
    elif args.interface:
        main_menu()


def main_menu() -> None:
    while True:
        print('Текущий каталог', os.getcwd())
        print(
            """
Выберите действие: 

0. Сменить рабочий каталог программы
1. Преобразовать PDF в Docx
2. Преобразовать Docx в PDF
3. Произвести сжатие изображений
4. Удалить группу файлов
5. Выход
            """)
        answer = input('Ваш выбор: ')
        match answer:
            case '0':
                os.chdir(get_directory())
            case '1':
                pdf_to_docx()
            case '2':
                docx_to_pdf()
            case '3':
                crop_image()
            case '4':
                delete_files()
            case '5':
                break
            case _:
                print('Неверный выбор!')


def delete_files() -> None:
    print("""
1. Удалить все файлы, начинающиеся на определенную подстроку
2. Удалить все файлы, заканчивающиеся на определенную подстроку
3. Удалить все файлы, содержащие определенную подстроку
4. Удалить все файлы по расширению
5. Выход
    """)
    while True:
        choice = input('Введите номер действия: ')
        match choice:
            case '1':
                delete_starts_with()
            case '2':
                delete_ends_with()
            case '3':
                delete_includes()
            case '4':
                delete_extension()
            case 5:
                break
            case _:
                print('Неверный выбор!')


def delete_extension() -> None:
    while True:
        files = os.listdir()
        file_found = False
        print_files(files)
        string = input('Введите подстроку (выйти - exit): ')
        if string == 'exit':
            break
        for file in files:
            if string == os.path.splitext(file)[1]:
                print(f'Удален файл {file}')
                os.remove(file)
                file_found = True
        if not file_found:
            print('Таких файлов не найдено!')


def delete_includes() -> None:
    while True:
        files = os.listdir()
        file_found = False
        print_files(files)
        string = input('Введите подстроку (выйти - exit): ')
        if string == 'exit':
            break
        for file in files:
            if string in os.path.splitext(file)[0]:
                print(f'Удален файл {file}')
                os.remove(file)
                file_found = True
        if not file_found:
            print('Таких файлов не найдено!')


def delete_ends_with() -> None:
    while True:
        files = os.listdir()
        file_found = False
        print_files(files)
        string = input('Введите подстроку (выйти - exit): ')
        if string == 'exit':
            break
        for file in files:
            if os.path.splitext(file)[0].endswith(string):
                print(f'Удален файл {file}')
                os.remove(file)
                file_found = True
        if not file_found:
            print('Таких файлов не найдено!')


def delete_starts_with() -> None:
    while True:
        files = os.listdir()
        file_found = False
        print_files(files)
        string = input('Введите подстроку (выйти - exit): ')
        if string == 'exit':
            break
        for file in files:
            if file.startswith(string):
                print(f'Удален файл {file}')
                os.remove(file)
                file_found = True
        if not file_found:
            print('Таких файлов не найдено!')


def pdf_to_docx() -> None:
    while True:
        files = os.listdir()
        print_files(files)
        try:
            choice = int(input('Выберите файл: '))
            if choice == len(files) - 1:
                break
            input_path = files[choice]
            if input_path.endswith('.pdf'):
                new_path = input_path.replace('.pdf', '.docx')
                parse(input_path, new_path)
            else:
                print('Это не PDF-файл!')
        except Exception as e:
            print(e)


def docx_to_pdf() -> None:
    while True:
        files = os.listdir()
        print_files(files)
        try:
            choice = int(input('Выберите файл: '))
            if choice == len(files) - 1:
                break
            input_path = files[choice]
            if input_path.endswith('.docx'):
                new_path = input_path.replace('.docx', '.pdf')
                convert(input_path, new_path)
            else:
                print('Это не Docx-файл!')
        except Exception as e:
            print(e)


def crop_image() -> None:
    while True:
        files = os.listdir()
        print_files(files)
        try:
            choice = int(input('Выберите файл: '))
            if choice == len(files) - 1:
                break
            image_name = files[choice]
            if (image_name.endswith('.jpg') or
                    image_name.endswith('.png') or
                    image_name.endswith('.jpeg')):
                try:
                    crop = int(input('Введите степень сжатия: '))
                    image_file = Image.open(image_name)
                    image_file.save(image_name, quality=100 - crop)
                    image_file.close()
                    print('Изображение сохранено!')
                except ValueError:
                    print('Введите целое число!')

            else:
                print('Это не изображение!')
        except Exception as e:
            print(e)


def print_files(files: list[str]) -> None:
    num = 0
    for file in files:
        print(f'{num}. {file}')
        num += 1
    print(f'{num}. Выход')


def get_directory() -> str | None:
    while True:
        dir = input('Введите путь к рабочему каталогу: ')
        if dir == 'exit':
            break
        else:
            if os.path.exists(dir) and os.path.isdir(dir):
                return dir
            else:
                print('По указанному пути нет директории!')


main()
# pdf_to_docx()
# compress_images()
