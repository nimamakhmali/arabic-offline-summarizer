# demo/app.py
"""
Arabic Offline Summarizer - Gradio Demo v2.0
Professional, RTL-ready, fully offline interface

Run: python demo/app.py
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Global State ─────────────────────────────────────────────────────────────
_summarizer = None
_load_error = None


def get_summarizer():
    """Lazy initialization of summarizer"""
    global _summarizer, _load_error
    
    if _summarizer is not None:
        return _summarizer, None

    if _load_error is not None:
        return None, _load_error

    try:
        from arabic_summarizer import ArabicSummarizer
        _summarizer = ArabicSummarizer(verbose=True)
        # Warmup
        _summarizer.warmup()
        return _summarizer, None
    except Exception as e:
        _load_error = str(e)
        logger.error(f"Failed to load summarizer: {e}")
        return None, str(e)


def load_model_btn_click():
    """Called when user clicks 'Load Model' button"""
    global _summarizer, _load_error
    _summarizer = None
    _load_error = None
    
    summarizer, error = get_summarizer()
    
    if error:
        return (
            f"❌ خطا در بارگذاری: {error}",
            gr.update(variant="secondary"),
        )

    info = summarizer.get_info()
    status = (
        f"✅ مدل آماده | "
        f"حالت: {info['mode'].upper()} | "
        f"نسخه: {info['version']}"
    )
    return status, gr.update(variant="primary", value="مدل بارگذاری شد ✓")


def count_words(text: str) -> str:
    """Count words in text for display"""
    if not text:
        return "۰ کلمه"
    count = len(text.split())
    return f"{count:,} کلمه"


def summarize_text(
    text: str,
    ratio: float,
    use_hybrid: bool,
    num_beams: int,
    use_quality_mode: bool,
) -> tuple:
    """
    Main summarization function called by Gradio.
    
    Returns:
        (summary_text, stats_text, word_count_text)
    """
    # Validate input
    if not text or not text.strip():
        return (
            "",
            "⚠️ لطفاً متنی وارد کنید",
            "۰ کلمه",
        )

    if len(text.split()) < 15:
        return (
            text.strip(),
            "⚠️ متن بسیار کوتاه است - بدون تغییر برگردانده شد",
            count_words(text),
        )

    # Get summarizer
    summarizer, error = get_summarizer()
    if error:
        return "", f"❌ مدل بارگذاری نشده: {error}", ""

    try:
        # Apply quality preset
        effective_beams = min(6, num_beams + 2) if use_quality_mode else num_beams

        result = summarizer.summarize(
            text=text,
            ratio=ratio,
            hybrid=use_hybrid,
            num_beams=effective_beams,
            return_stats=True,
        )

        summary = result["summary"]
        
        # Build stats string
        stats = (
            f"⏱ زمان: {result['time_seconds']}s | "
            f"📊 فشرده‌سازی: {result['compression_ratio']:.1%} | "
            f"📝 ورودی: {result['input_words']} کلمه | "
            f"✏️ خروجی: {result['output_words']} کلمه | "
            f"🔧 حالت: {result['mode'].upper()}"
        )

        return summary, stats, count_words(summary)

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return "", f"❌ خطا: {str(e)}", ""


def evaluate_summary(original: str, summary: str) -> str:
    """Compute ROUGE scores between original and summary"""
    if not original or not summary:
        return "⚠️ متن اصلی و خلاصه هر دو باید وارد شوند"
    
    try:
        from arabic_summarizer import ArabicSummarizationEvaluator
        evaluator = ArabicSummarizationEvaluator()
        result = evaluator.evaluate_single(original, summary)
        
        output = f"📏 نسبت فشرده‌سازی: {result['compression_ratio']}\n"
        output += f"📝 طول اصلی: {result['original_length']} کلمه\n"
        output += f"✏️ طول خلاصه: {result['generated_length']} کلمه\n"
        
        if "rouge1" in result:
            output += f"\nROUGE-1: {result['rouge1']:.4f}\n"
            output += f"ROUGE-2: {result['rouge2']:.4f}\n"
            output += f"ROUGE-L: {result['rougeL']:.4f}"
        
        return output
    except Exception as e:
        return f"خطا در ارزیابی: {str(e)}"


# ─── Sample Texts ─────────────────────────────────────────────────────────────

SAMPLE_TEXTS = {
    "📰 خبر": (
        "أعلنت منظمة الصحة العالمية عن إطلاق مبادرة عالمية جديدة تهدف إلى "
        "تعزيز التغطية الصحية الشاملة في الدول النامية. وتشمل هذه المبادرة "
        "توفير اللقاحات الأساسية والرعاية الصحية الأولية لما يزيد على مليار "
        "شخص حول العالم. وأكد المدير العام للمنظمة أن هذه الخطوة تمثل نقلة "
        "نوعية في مسيرة تحقيق أهداف التنمية المستدامة المتعلقة بالصحة. "
        "وستعمل المنظمة بالتعاون مع الحكومات والقطاع الخاص ومنظمات المجتمع "
        "المدني لضمان وصول الخدمات الصحية إلى الفئات الأكثر احتياجاً في "
        "المناطق النائية والمجتمعات المهمشة. وتأتي هذه المبادرة في سياق "
        "الجهود العالمية الرامية إلى تحقيق العدالة الصحية وسد الفجوات "
        "القائمة بين الدول المتقدمة والنامية في مجال الرعاية الصحية."
    ),
    "🔬 علمي": (
        "يُعدّ الذكاء الاصطناعي من أبرز الثورات التكنولوجية في القرن الحادي "
        "والعشرين، إذ يُحاكي القدرات المعرفية البشرية من خلال خوارزميات "
        "معقدة وشبكات عصبية اصطناعية متطورة. ويشمل الذكاء الاصطناعي "
        "مجالات متعددة كتعلم الآلة والتعلم العميق ومعالجة اللغة الطبيعية "
        "والرؤية الحاسوبية والروبوتيات. وتتجلى تطبيقاته في مختلف القطاعات "
        "الحيوية كالطب والتعليم والصناعة والنقل والزراعة. ولا يزال الباحثون "
        "يسعون إلى تطوير نماذج أكثر كفاءة وأقل استهلاكاً للطاقة، مع الحرص "
        "على معالجة الإشكاليات الأخلاقية المرتبطة باستخدام هذه التقنيات، "
        "كمسائل الخصوصية والتحيز الخوارزمي وتأثيرها على سوق العمل."
    ),
    "📖 قرآني": (
        "قال الله تعالى في كتابه الكريم: إن الله يأمر بالعدل والإحسان "
        "وإيتاء ذي القربى وينهى عن الفحشاء والمنكر والبغي يعظكم لعلكم "
        "تذكرون. وقد أكد علماء التفسير أن هذه الآية الكريمة تجمع في ألفاظ "
        "قليلة أسس الشريعة الإسلامية كلها، إذ تأمر بثلاثة أشياء وتنهى "
        "عن ثلاثة أشياء. فالعدل هو إعطاء كل ذي حق حقه، والإحسان هو "
        "التفضل على الناس بما يزيد على الواجب، وإيتاء ذي القربى هو "
        "صلة الرحم والإنفاق على الأقارب."
    ),
}


# ─── CSS ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* RTL support for Arabic text areas */
.arabic-text textarea {
    direction: rtl !important;
    font-family: 'Arial', 'Tahoma', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.8 !important;
}

/* Stats box styling */
.stats-box {
    background: #f0f4f8;
    border-radius: 8px;
    padding: 8px;
    font-size: 13px;
    color: #444;
}

/* Header styling */
.header-box {
    background: linear-gradient(135deg, #1a5276, #2980b9);
    color: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 15px;
}

/* Primary button */
.primary-btn {
    background: #2980b9 !important;
    font-size: 16px !important;
    font-weight: bold !important;
}

/* Word count badge */
.word-count {
    font-size: 12px;
    color: #777;
    text-align: right;
}
"""


# ─── Build Interface ───────────────────────────────────────────────────────────

def build_interface():
    """Build and return the Gradio interface"""

    with gr.Blocks(
        title="خلاصه‌ساز عربی آفلاین | Arabic Offline Summarizer",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css=CUSTOM_CSS,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <h1>📖 خلاصه‌ساز هوشمند متون عربی</h1>
            <h3>Arabic Offline Summarizer v2.0</h3>
            <p>🔌 کاملاً آفلاین &nbsp;|&nbsp; ⚡ سریع (۵-۱۵ ثانیه) &nbsp;|&nbsp; 
               🎯 دقیق (NLP عربی) &nbsp;|&nbsp; 🛡️ بدون ارسال داده</p>
        </div>
        """)

        # ── Model Status Bar ───────────────────────────────────────────────
        with gr.Row():
            model_status = gr.Textbox(
                label="وضعیت مدل",
                value="⏳ کلیک کنید برای بارگذاری مدل...",
                interactive=False,
                scale=4,
            )
            load_btn = gr.Button(
                "🔄 بارگذاری مدل",
                variant="secondary",
                scale=1,
                min_width=150,
            )

        gr.Divider()

        # ── Main Area ─────────────────────────────────────────────────────
        with gr.Row(equal_height=True):

            # ── Left Column: Input ─────────────────────────────────────
            with gr.Column(scale=5):
                gr.Markdown("### 📝 متن ورودی")

                input_text = gr.Textbox(
                    label="",
                    placeholder="متن عربی خود را اینجا بنویسید یا بچسبانید...\n"
                                "النص العربي هنا...",
                    lines=14,
                    max_lines=25,
                    elem_classes=["arabic-text"],
                    show_copy_button=True,
                )

                input_word_count = gr.Markdown(
                    "۰ کلمه",
                    elem_classes=["word-count"],
                )

                # ── Sample Buttons ──────────────────────────────────────
                gr.Markdown("**نمونه‌های آماده:**")
                with gr.Row():
                    for label, sample_text in SAMPLE_TEXTS.items():
                        gr.Button(label, size="sm").click(
                            fn=lambda t=sample_text: t,
                            outputs=input_text,
                        )

            # ── Right Column: Controls + Output ────────────────────────
            with gr.Column(scale=5):
                gr.Markdown("### ⚙️ تنظیمات")

                with gr.Row():
                    ratio_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.30,
                        value=0.20,
                        step=0.05,
                        label="نسبت خلاصه (۱۰٪-۳۰٪)",
                        info="درصد طول خلاصه نسبت به متن اصلی",
                    )

                with gr.Accordion("🔬 تنظیمات پیشرفته", open=False):
                    use_hybrid = gr.Checkbox(
                        value=True,
                        label="روش ترکیبی (Hybrid: استخراجی + انتزاعی)",
                        info="برای متون طولانی پیشنهاد می‌شود"
                    )
                    num_beams = gr.Slider(
                        minimum=2,
                        maximum=6,
                        value=4,
                        step=1,
                        label="عمق جستجو (Beam Search)",
                        info="بالاتر = بهتر اما کندتر"
                    )
                    use_quality = gr.Checkbox(
                        value=False,
                        label="حالت کیفیت بالا (کندتر)",
                    )

                # ── Summarize Button ────────────────────────────────────
                summarize_btn = gr.Button(
                    "🚀 خلاصه‌سازی",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"],
                )

                gr.Markdown("### 📋 خلاصه تولید شده")
                
                output_text = gr.Textbox(
                    label="",
                    lines=10,
                    max_lines=20,
                    elem_classes=["arabic-text"],
                    show_copy_button=True,
                    interactive=False,
                    placeholder="خلاصه اینجا نمایش داده می‌شود..."
                )

                output_word_count = gr.Markdown(
                    "",
                    elem_classes=["word-count"],
                )

                stats_display = gr.Textbox(
                    label="📊 آمار عملکرد",
                    interactive=False,
                    lines=2,
                    elem_classes=["stats-box"],
                )

        # ── Evaluation Tab ─────────────────────────────────────────────────
        with gr.Accordion("🔍 ارزیابی کیفیت (ROUGE)", open=False):
            gr.Markdown(
                "برای ارزیابی خودکار، خلاصه مرجع (reference) وارد کنید:"
            )
            with gr.Row():
                ref_text = gr.Textbox(
                    label="خلاصه مرجع (Reference Summary)",
                    lines=4,
                    elem_classes=["arabic-text"],
                    placeholder="خلاصه انسانی یا مرجع را اینجا وارد کنید..."
                )
                eval_output = gr.Textbox(
                    label="نتایج ارزیابی",
                    lines=4,
                    interactive=False,
                )
            eval_btn = gr.Button("محاسبه ROUGE", variant="secondary")

        # ── Footer ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; margin-top:20px; color:#888; font-size:13px;">
            <p>
                Arabic Offline Summarizer v2.0 &nbsp;|&nbsp;
                Powered by AraT5v2 + ONNX Runtime &nbsp;|&nbsp;
                🔌 Fully Offline &nbsp;|&nbsp;
                Apache 2.0 License
            </p>
        </div>
        """)

        # ─── Event Handlers ───────────────────────────────────────────────

        # Load model
        load_btn.click(
            fn=load_model_btn_click,
            outputs=[model_status, load_btn],
        )

        # Live word count
        input_text.change(
            fn=count_words,
            inputs=input_text,
            outputs=input_word_count,
        )

        # Summarize
        summarize_btn.click(
            fn=summarize_text,
            inputs=[
                input_text,
                ratio_slider,
                use_hybrid,
                num_beams,
                use_quality,
            ],
            outputs=[output_text, stats_display, output_word_count],
        )

        # Evaluate
        eval_btn.click(
            fn=evaluate_summary,
            inputs=[input_text, output_text],
            outputs=eval_output,
        )

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        favicon_path=None,
    )