# -*- mode: python ; coding: utf-8 -*-

pdf2docx_root = 'C:\\Users\\Rafael\\Desktop\\GitProjects\\University\\Tasks\\New_Console_Tweaks\\.venv\\Lib\\site-packages\\pdf2docx'  # укажите абсолютный путь к корню pdf2docx в вашей системе

datas = []
for root, dirs, files in os.walk(pdf2docx_root):
    for f in files:
        full_path = os.path.join(root, f)
        relative_path = os.path.relpath(root, pdf2docx_root)
        datas.append((full_path, os.path.join('pdf2docx', relative_path)))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['pdf2docx', 'pymupdf'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Office_Tweaks_1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
