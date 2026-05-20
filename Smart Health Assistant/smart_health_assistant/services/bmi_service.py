def bmi_category(bmi: float) -> str:
    """Map BMI to WHO-style category labels."""
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal weight"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def bmi_recommendations(category: str) -> list[str]:
    """Evidence-informed general wellness tips keyed by BMI category (not medical advice)."""
    base = [
        "Discuss significant diet or exercise changes with a qualified clinician if you have "
        "medical conditions or take medications.",
        "Prioritize consistent sleep and hydration alongside nutrition and activity.",
    ]
    if category == "Underweight":
        return base + [
            "Focus on nutrient-dense meals with adequate protein and healthy fats.",
            "Consider strength training to support lean mass with professional guidance.",
        ]
    if category == "Normal weight":
        return base + [
            "Maintain balanced meals and regular physical activity you can sustain.",
            "Use periodic check-ins (energy, strength, mood) to adjust habits gradually.",
        ]
    if category == "Overweight":
        return base + [
            "A modest, sustainable calorie deficit combined with higher protein often helps fat loss.",
            "Add daily movement (walking) plus 2–3 resistance sessions per week when appropriate.",
        ]
    return base + [
        "Work with a clinician for individualized targets; gradual, consistent changes matter most.",
        "Emphasize whole foods, portion awareness, and joint-friendly activity as tolerated.",
    ]


def format_bmi_insights(bmi: float, category: str) -> str:
    lines = [
        f"BMI: {bmi:.2f}",
        f"Category: {category}",
        "Recommendations:",
        *[f"- {r}" for r in bmi_recommendations(category)],
    ]
    return "\n".join(lines)
