from smart_health_assistant.enums import Gender
from smart_health_assistant.models import UserHealthProfile


def _intensity_note(age: int) -> str:
    if age >= 65:
        return (
            "Emphasize balance, mobility, and low-impact cardio; obtain clearance if new to exercise."
        )
    if age >= 45:
        return "Include warm-ups and recovery days; progress volume gradually."
    return "Progress load weekly by small increments; stop if sharp pain occurs."


def build_weekly_workout(profile: UserHealthProfile, meal_plan_summary: str) -> str:
    """Seven-day schedule using age, gender, and diet-plan context."""
    age = profile.age
    gender = profile.gender
    intensity = _intensity_note(age)

    gender_line = (
        "Training plan is appropriate for any sex; adjust loads to individual recovery."
        if gender is Gender.OTHER
        else f"Tailored for typical recovery patterns; listen to your body ({gender.value})."
    )

    lines = [
        "### Weekly workout schedule",
        gender_line,
        f"Age context: {age} — {intensity}",
        "",
        "| Day | Focus | Details |",
        "|-----|-------|---------|",
        "| Mon | Strength (full body) | Squat or leg press pattern, row, push, hinge; 3x8-12 |",
        "| Tue | Zone 2 cardio | 30-40 min brisk walk, cycle, or elliptical |",
        "| Wed | Active recovery | 20-30 min easy walk + mobility drills |",
        "| Thu | Strength (upper/core) | Press, pull, carries, planks; 3x8-12 |",
        "| Fri | Intervals (optional) | 8-10 x 1 min hard / 1 min easy if joints tolerate |",
        "| Sat | Long easy cardio or sport | 45-60 min conversational pace |",
        "| Sun | Rest or gentle yoga | Stretching, breathing, light steps |",
        "",
        "_Alignment with meal plan:_ fuel workouts with a balanced meal 2-3 hours before "
        "or a small carb + protein snack closer to session time.",
        "",
        "Diet planner context (for continuity):",
        meal_plan_summary[:800] + ("…" if len(meal_plan_summary) > 800 else ""),
    ]
    return "\n".join(lines)
