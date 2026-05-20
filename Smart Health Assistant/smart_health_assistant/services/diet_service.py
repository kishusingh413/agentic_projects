from smart_health_assistant.enums import DietaryPreference
from smart_health_assistant.models import UserHealthProfile


def build_meal_plan(profile: UserHealthProfile, bmi_category_label: str) -> str:
    """Rule-based meal outline from dietary preference and BMI context."""
    pref = profile.dietary_preference
    lines = [
        "### Tailored meal plan (template)",
        f"Dietary preference: {pref.value}",
        f"BMI context: {bmi_category_label}",
        "",
    ]

    if pref is DietaryPreference.VEG:
        lines += [
            "**Breakfast:** Oats with milk or fortified soy yogurt, fruit, nuts or seeds.",
            "**Lunch:** Dal + brown rice or roti, mixed vegetable sabzi, cucumber salad.",
            "**Dinner:** Chickpea or paneer curry (portion-aware), quinoa or millet, greens.",
            "**Snacks:** Fruit, roasted chana, or hummus with vegetables.",
        ]
    elif pref is DietaryPreference.NON_VEG:
        lines += [
            "**Breakfast:** Eggs or Greek yogurt, whole-grain toast, fruit.",
            "**Lunch:** Grilled chicken or fish, quinoa or rice, large salad or steamed vegetables.",
            "**Dinner:** Lean protein (fish/tofu/chicken), sweet potato or whole grains, broccoli.",
            "**Snacks:** Cottage cheese, edamame, or a small handful of nuts.",
        ]
    else:
        lines += [
            "**Breakfast:** Tofu scramble with vegetables, whole-grain toast, berries.",
            "**Lunch:** Lentil bowl with tahini dressing, farro or brown rice, leafy salad.",
            "**Dinner:** Tempeh or bean chili, roasted vegetables, quinoa.",
            "**Snacks:** Hummus with carrots, mixed nuts (if tolerated), or fruit.",
        ]

    if bmi_category_label in {"Overweight", "Obese"}:
        lines.append("")
        lines.append(
            "_Portion focus:_ emphasize vegetables and lean proteins; keep oils and "
            "refined carbs moderate unless your clinician advises otherwise."
        )
    elif bmi_category_label == "Underweight":
        lines.append("")
        lines.append(
            "_Energy focus:_ add healthy extras (nuts, avocado, extra legumes) between meals "
            "if appetite allows."
        )

    return "\n".join(lines)
