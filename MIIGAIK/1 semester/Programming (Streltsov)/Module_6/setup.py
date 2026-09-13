from cx_Freeze import setup, Executable
import os

# Путь к корню pdf2docx в вашем виртуальном окружении
pdf2docx_path = r"C:\Users\Rafael\Desktop\GitProjects\University\Tasks\New_Console_Tweaks\.venv\Lib\site-packages\pdf2docx"

# Укажите все дополнительные файлы и папки, которые нужно включить
build_exe_options = {
    "packages": ["pdf2docx", "pymupdf"],  # импортируемые пакеты
    "include_files": [
        (pdf2docx_path, "pdf2docx")  # копируем всю папку pdf2docx
    ],
    "excludes": []
}

setup(
    name="Office_Tweaks_1",
    version="1.0",
    description="My pdf2docx app",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py", target_name="Office_Tweaks_1.exe", base=None)],
)
