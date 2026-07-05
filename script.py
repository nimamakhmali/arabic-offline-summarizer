# setup_project.py
"""
اجرا کنید تا کل ساختار پروژه به‌صورت خودکار ساخته شود:
    python setup_project.py
"""

from pathlib import Path

BASE = Path("arabic_offline_summarizer")

# ─── تعریف کامل ساختار ───────────────────────────────────────────────────────

STRUCTURE = {
    # کتابخانه اصلی
    "src/arabic_summarizer/__init__.py": "",
    "src/arabic_summarizer/core.py": "",
    "src/arabic_summarizer/config.py": "",
    "src/arabic_summarizer/preprocessor.py": "",
    "src/arabic_summarizer/postprocessor.py": "",
    "src/arabic_summarizer/extractive.py": "",
    "src/arabic_summarizer/chunker.py": "",
    "src/arabic_summarizer/evaluate.py": "",
    "src/arabic_summarizer/cli.py": "",

    # دمو
    "demo/app.py": "",

    # اسکریپت‌ها
    "scripts/download_model.py": "",
    "scripts/export_to_onnx.py": "",

    # تست‌ها
    "tests/__init__.py": "",
    "tests/test_summarizer.py": "",
    "tests/test_evaluate.py": "",
    "tests/test_integration.py": "",

    # مدل‌ها
    "models/.gitkeep": "",

    # داده‌ها
    "data/samples/create_samples.py": "",
    "data/samples/test_samples.json": "[]",

    # مستندات
    "docs/TECHNICAL_REPORT.md": "# گزارش فنی\n",

    # ریشه
    "run_demo.py": "",
    "README.md": "# Arabic Offline Summarizer\n",
    "requirements.txt": "",
    "pyproject.toml": "",
    "Makefile": "",
    ".gitignore": "",
}


def create_structure():
    print(f"\n🏗️  ساخت ساختار پروژه در: {BASE.absolute()}\n")

    created_dirs = set()
    created_files = []

    for relative_path, content in STRUCTURE.items():
        full_path = BASE / relative_path

        # ساخت پوشه
        if full_path.parent not in created_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            created_dirs.add(full_path.parent)
            print(f"  📁 {full_path.parent.relative_to(BASE)}/")

        # ساخت فایل
        if not full_path.exists():
            full_path.write_text(content, encoding="utf-8")
            created_files.append(relative_path)
            print(f"     📄 {full_path.name}")

    print(f"\n✅ ساختار پروژه آماده شد!")
    print(f"   📁 پوشه‌ها: {len(created_dirs)}")
    print(f"   📄 فایل‌ها: {len(created_files)}")
    print(f"\n🚀 مرحله بعدی:")
    print(f"   cd {BASE.name}")
    print(f"   pip install -r requirements.txt")
    print(f"   python scripts/download_model.py\n")


if __name__ == "__main__":
    create_structure()