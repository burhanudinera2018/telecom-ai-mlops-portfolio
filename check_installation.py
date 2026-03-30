#!/usr/bin/env python
"""Script untuk memverifikasi instalasi packages"""

import sys
import importlib

# Daftar packages yang harus diinstall
packages = {
    'numpy': '1.24.3',
    'pandas': '2.0.3',
    'sklearn': '1.3.0',
    'scipy': '1.10.1',
    'matplotlib': '3.7.2',
    'seaborn': '0.12.2',
    'imblearn': '0.11.0',
    'shap': '0.42.1',
    'joblib': '1.3.2',
    'pytest': '7.4.0',
    'yaml': '6.0.1',
    'tqdm': '4.65.0'
}

# Mapping nama module ke nama import
import_names = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'sklearn',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'imblearn': 'imblearn',
    'shap': 'shap',
    'joblib': 'joblib',
    'pytest': 'pytest',
    'yaml': 'yaml',
    'tqdm': 'tqdm'
}

print("=" * 60)
print(f"Python Version: {sys.version}")
print("=" * 60)
print("\n📦 Package Installation Status:\n")

successful = []
failed = []

for package, expected_version in packages.items():
    import_name = import_names[package]
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'version unknown')
        
        # Cek versi
        if version == expected_version or expected_version in version:
            status = "✅"
            successful.append(package)
        else:
            status = "⚠️"
            print(f"  {status} {package:15} installed: {version:15} (expected: {expected_version})")
            successful.append(package)  # tetap anggap sukses meski versi beda
            continue
            
        print(f"  {status} {package:15} {version:15}")
        successful.append(package)
        
    except ImportError as e:
        status = "❌"
        print(f"  {status} {package:15} FAILED TO INSTALL")
        print(f"       Error: {str(e)}")
        failed.append(package)

print("\n" + "=" * 60)
print(f"📊 Summary:")
print(f"   ✅ Successfully installed: {len(successful)} packages")
print(f"   ❌ Failed to install: {len(failed)} packages")

if failed:
    print(f"\n❌ Failed packages: {', '.join(failed)}")
    
print("\n💡 Next steps:")
if failed:
    print("   1. Run troubleshooting for failed packages")
    print("   2. Check error messages above")
    print("   3. Try installing failed packages individually")
else:
    print("   ✅ All packages installed successfully!")
    print("   You can proceed with project setup")

print("=" * 60)