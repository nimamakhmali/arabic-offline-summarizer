# arabic-offline-summarizer
An offline, high-accuracy Arabic text summarization engine optimized for real-world applications, fast local execution, and complete internet independence.


# 📖 Arabic Offline Summarizer (AOS) v2.0

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-Apache%202.0-orange)
![Offline](https://img.shields.io/badge/mode-100%25%20Offline-success)
![Arabic](https://img.shields.io/badge/language-Arabic%20MSA-red)

**خلاصه‌ساز هوشمند متون عربی - کاملاً آفلاین، سریع، دقیق**

*Intelligent Arabic Text Summarization - Fully Offline, Fast, Accurate*

</div>

---

## ✨ ویژگی‌های کلیدی

| ویژگی | جزئیات |
|-------|---------|
| 🔌 **کاملاً آفلاین** | بدون نیاز به اینترنت پس از نصب |
| ⚡ **سریع** | ۵-۱۵ ثانیه برای متون معمولی |
| 🎯 **دقیق** | خلاصه ۱۰-۳۰٪ با حفظ مفاهیم اصلی |
| 🔄 **Hybrid** | ترکیب استخراجی + انتزاعی |
| 📦 **ONNX** | مدل بهینه و کوانتیزه‌شده |
| 🌐 **چندکاربردی** | قرآن، خبر، سند، پژوهش |

---

## 🚀 نصب سریع

```bash
# ۱. کلون پروژه
git clone https://github.com/your-org/arabic-offline-summarizer.git
cd arabic-offline-summarizer

# ۲. محیط مجازی
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows

# ۳. نصب وابستگی‌ها
pip install -r requirements.txt

# ۴. دانلود مدل (یک‌بار - نیاز به اینترنت)
python scripts/download_model.py

# ۵. تست سریع
python run_demo.py --quick
```

---

## 📋 استفاده

### Python API

```python
from arabic_summarizer import ArabicSummarizer

# ایجاد نمونه
summarizer = ArabicSummarizer()

# خلاصه‌سازی ساده
text = """
أعلنت منظمة الصحة العالمية عن إطلاق مبادرة عالمية جديدة تهدف إلى تعزيز 
التغطية الصحية الشاملة في الدول النامية. وتشمل هذه المبادرة توفير اللقاحات 
الأساسية والرعاية الصحية الأولية لما يزيد على مليار شخص حول العالم.
"""

summary = summarizer.summarize(text, ratio=0.2)
print(summary)

# با آمار کامل
result = summarizer.summarize(text, ratio=0.2, return_stats=True)
print(f"خلاصه: {result['summary']}")
print(f"زمان: {result['time_seconds']}s")
print(f"نسبت: {result['compression_ratio']:.1%}")
print(f"حالت: {result['mode']}")

# پردازش دسته‌ای
texts = [text1, text2, text3]
results = summarizer.batch_summarize(texts, ratio=0.15)
```

### تنظیمات پیشرفته

```python
from arabic_summarizer import ArabicSummarizer, SummarizerConfig, QURAN_CONFIG

# برای متون قرآنی (حفظ تشکیل)
summarizer = ArabicSummarizer(config=QURAN_CONFIG)

# تنظیمات سفارشی
config = SummarizerConfig()
config.generation.num_beams = 6          # کیفیت بالاتر
config.generation.default_ratio = 0.15  # خلاصه‌تر
config.hybrid.enabled = True             # ترکیبی
summarizer = ArabicSummarizer(config=config)

# حالت ONNX (سریع‌تر)
summarizer = ArabicSummarizer(
    model_path="models/araT5_summarizer_onnx_quantized",
    force_mode="onnx"
)
```

---

## 🏗️ معماری سیستم

```
متن ورودی (عربی)
        │
        ▼
┌───────────────────┐
│   Preprocessor    │  ← نرمال‌سازی، پاکسازی، تشکیل
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Extractive Layer │  ← TF-IDF + Position (برای متون طولانی)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  AraT5v2 (ONNX)   │  ← تولید خلاصه انتزاعی
│  INT8 Quantized   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Postprocessor    │  ← حذف تکرار، اصلاح نگارش
└───────────────────┘
        │
        ▼
   خلاصه نهایی
```

---

## 📁 ساختار پروژه

```
arabic_offline_summarizer/
├── src/arabic_summarizer/
│   ├── __init__.py          ← Public API
│   ├── core.py              ← موتور اصلی
│   ├── preprocessor.py      ← پیش‌پردازش عربی
│   ├── postprocessor.py     ← پس‌پردازش
│   ├── extractive.py        ← خلاصه‌سازی استخراجی
│   ├── chunker.py           ← مدیریت متون طولانی
│   ├── evaluate.py          ← ارزیابی ROUGE + کیفیت
│   └── config.py            ← تنظیمات
├── demo/
│   └── app.py               ← رابط Gradio
├── scripts/
│   ├── download_model.py    ← دانلود مدل
│   └── export_to_onnx.py    ← بهینه‌سازی ONNX
├── tests/
│   └── test_summarizer.py   ← تست‌های واحد
├── models/                  ← مدل‌های دانلودشده
├── run_demo.py              ← نقطه شروع سریع
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚡ بهینه‌سازی برای تولید (ONNX)

```bash
# مرحله ۱: دانلود مدل
python scripts/download_model.py

# مرحله ۲: تبدیل به ONNX + کوانتیزاسیون
python scripts/export_to_onnx.py

# مرحله ۳: بنچمارک
python scripts/export_to_onnx.py --benchmark

# استفاده از مدل بهینه
python run_demo.py --mode onnx
```

---

## 🖥️ رابط گرافیکی

```bash
python demo/app.py
# باز می‌شود در: http://localhost:7860
```

---

## 📊 ارزیابی

```python
from arabic_summarizer import ArabicSummarizationEvaluator

evaluator = ArabicSummarizationEvaluator()

# ارزیابی تکی
result = evaluator.evaluate_single(
    original=original_text,
    generated=summary,
    reference=human_summary  # اختیاری
)
print(f"کیفیت: {result['quality_score']}/100")
print(f"ROUGE-1: {result.get('rouge1', 'N/A')}")

# بنچمارک کامل
report = evaluator.benchmark_summarizer(summarizer, test_cases)
```

---

## 🧪 تست‌ها

```bash
# همه تست‌ها
pytest tests/ -v

# فقط تست‌های آفلاین (بدون مدل)
pytest tests/ -v -m offline

# با پوشش کد
pytest tests/ --cov=arabic_summarizer --cov-report=html
```

---

## 📦 کاربردها

- **🕌 قرآن و تفسیر**: حفظ تشکیل + خلاصه دقیق
- **📰 رسانه و خبر**: خلاصه سریع اخبار عربی
- **📚 پژوهش**: خلاصه مقالات و پژوهش‌های علمی
- **📋 اسناد رسمی**: خلاصه قراردادها و بخشنامه‌ها
- **🎓 آموزش**: خلاصه مطالب درسی

---

## 📜 مجوز

Apache License 2.0 — آزاد برای استفاده تجاری و غیرتجاری

---

## 📞 تماس

برای همکاری و سرمایه‌گذاری: [ایمیل یا تلگرام]