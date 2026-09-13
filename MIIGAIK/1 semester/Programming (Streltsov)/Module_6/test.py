import os
from docx2pdf import convert

my_path = r"C:\Users\Rafael\Desktop\Student\Практика"
print(os.listdir(my_path))
print(os.path.isdir(my_path))
for file_name in os.listdir(my_path):
    if file_name.endswith(".docx"):
        convert(my_path+'\\'+file_name, file_name+'.pdf')