# data/samples/create_samples.py
"""
Create sample test data for evaluation
Run: python data/samples/create_samples.py
"""

import json
from pathlib import Path

SAMPLES = [
    {
        "id": "news_001",
        "domain": "news",
        "text": (
            "أعلنت منظمة الصحة العالمية عن إطلاق مبادرة عالمية جديدة تهدف إلى "
            "تعزيز التغطية الصحية الشاملة في الدول النامية. وتشمل هذه المبادرة "
            "توفير اللقاحات الأساسية والرعاية الصحية الأولية لما يزيد على مليار "
            "شخص حول العالم. وأكد المدير العام للمنظمة أن هذه الخطوة تمثل نقلة "
            "نوعية في مسيرة تحقيق أهداف التنمية المستدامة المتعلقة بالصحة."
        ),
        "reference_summary": (
            "أطلقت منظمة الصحة العالمية مبادرة لتعزيز التغطية الصحية "
            "في الدول النامية تشمل تقديم اللقاحات والرعاية الأولية لمليار شخص."
        ),
        "target_ratio": 0.20,
    },
    {
        "id": "scientific_001",
        "domain": "scientific",
        "text": (
            "يُعدّ الذكاء الاصطناعي من أبرز الثورات التكنولوجية في القرن الحادي "
            "والعشرين، إذ يُحاكي القدرات المعرفية البشرية من خلال خوارزميات معقدة. "
            "ويشمل الذكاء الاصطناعي مجالات متعددة كتعلم الآلة والتعلم العميق "
            "ومعالجة اللغة الطبيعية والرؤية الحاسوبية. وتتجلى تطبيقاته في "
            "مختلف القطاعات الحيوية كالطب والتعليم والصناعة والنقل والزراعة."
        ),
        "reference_summary": (
            "الذكاء الاصطناعي ثورة تقنية تحاكي القدرات البشرية "
            "وتشمل التعلم الآلي ومعالجة اللغة والرؤية الحاسوبية "
            "مع تطبيقات واسعة في الطب والتعليم والصناعة."
        ),
        "target_ratio": 0.20,
    },
    {
        "id": "quran_001",
        "domain": "religious",
        "text": (
            "قال الله تعالى في كتابه الكريم: إن الله يأمر بالعدل والإحسان "
            "وإيتاء ذي القربى وينهى عن الفحشاء والمنكر والبغي يعظكم لعلكم "
            "تذكرون. وقد أكد علماء التفسير أن هذه الآية الكريمة تجمع في ألفاظ "
            "قليلة أسس الشريعة الإسلامية، إذ تأمر بثلاثة وتنهى عن ثلاثة. "
            "فالعدل هو إعطاء كل ذي حق حقه، والإحسان هو التفضل على الناس."
        ),
        "reference_summary": (
            "آية قرآنية تأمر بالعدل والإحسان وصلة الرحم وتنهى "
            "عن الفحشاء والمنكر والبغي، وهي تجمع أسس الشريعة الإسلامية."
        ),
        "target_ratio": 0.15,
    },
    {
        "id": "official_001",
        "domain": "official",
        "text": (
            "قرار رقم 2024/أ: استناداً إلى أحكام القانون رقم 12 لسنة 2020، "
            "تقرر ما يلي: أولاً: إلزام مزودي خدمات الإنترنت بتوفير سرعات "
            "لا تقل عن مئة ميغابت في المناطق الحضرية. ثانياً: تخصيص ميزانية "
            "خمسة مليارات ريال لتطوير البنية الرقمية في المناطق الريفية "
            "خلال 2024-2026. ثالثاً: إنشاء هيئة تنظيمية للإشراف على التطبيق."
        ),
        "reference_summary": (
            "قرار حكومي يلزم مزودي الإنترنت بسرعات 100 ميغابت، "
            "ويخصص 5 مليارات ريال للمناطق الريفية، وينشئ هيئة تنظيمية."
        ),
        "target_ratio": 0.20,
    },
]


def main():
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    output_path = output_dir / "test_samples.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLES, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(SAMPLES)} samples to: {output_path}")

    # Save as plain text for manual inspection
    for sample in SAMPLES:
        txt_path = output_dir / f"{sample['id']}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"ID: {sample['id']}\n")
            f.write(f"Domain: {sample['domain']}\n")
            f.write(f"Target Ratio: {sample['target_ratio']}\n")
            f.write("\n--- TEXT ---\n")
            f.write(sample["text"])
            f.write("\n\n--- REFERENCE ---\n")
            f.write(sample["reference_summary"])
            f.write("\n")

    print(f"✓ Saved individual text files")


if __name__ == "__main__":
    main()