import json
from typing import Any

from smart_health_assistant.services.bmi_service import bmi_category


def calculate_bmi(height_cm: float, weight_kg: float) -> str:
    """
    Calculate Body Mass Index (BMI) from height in centimeters and weight in kilograms.

    Converts height to meters (height_m = height_cm / 100), then computes:
    BMI = weight_kg / (height_m ** 2).

    Returns a JSON string with numeric BMI, category, height_m, and inputs for downstream agents.
    """
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m**2)
    category = bmi_category(bmi)
    payload: dict[str, Any] = {
        "height_cm": height_cm,
        "height_m": round(height_m, 4),
        "weight_kg": weight_kg,
        "bmi": round(bmi, 2),
        "category": category,
    }
    return json.dumps(payload)
