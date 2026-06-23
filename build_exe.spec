# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['run_app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('app.py', '.'),
        ('main.py', '.'),
        ('blurrer.py', '.'),
        ('detector.py', '.'),
        ('tracker.py', '.'),
        ('preview.py', '.'),
        ('utils.py', '.'),
        ('yolov8n-face.pt', '.'),
    ],
    hiddenimports=['streamlit.web.bootstrap'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BlurrySlurry',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
)