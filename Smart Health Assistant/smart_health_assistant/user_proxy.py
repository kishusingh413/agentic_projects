import logging
from typing import Callable

from pydantic import ValidationError

from smart_health_assistant.enums import DietaryPreference, Gender
from smart_health_assistant.models import UserHealthProfile

logger = logging.getLogger(__name__)


def _parse_float(label: str, raw: str) -> float:
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a number.") from exc


def _parse_int(label: str, raw: str) -> int:
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a whole number.") from exc


def _parse_gender(raw: str) -> Gender:
    key = raw.strip().lower()
    mapping: dict[str, Gender] = {
        "m": Gender.MALE,
        "male": Gender.MALE,
        "man": Gender.MALE,
        "f": Gender.FEMALE,
        "female": Gender.FEMALE,
        "woman": Gender.FEMALE,
        "o": Gender.OTHER,
        "other": Gender.OTHER,
        "non-binary": Gender.OTHER,
        "nonbinary": Gender.OTHER,
    }
    if key not in mapping:
        raise ValueError("Gender must be male, female, or other (or m/f/o).")
    return mapping[key]


def _parse_diet(raw: str) -> DietaryPreference:
    key = raw.strip().lower().replace(" ", "-")
    mapping: dict[str, DietaryPreference] = {
        "1": DietaryPreference.VEG,
        "2": DietaryPreference.NON_VEG,
        "3": DietaryPreference.VEGAN,
        "veg": DietaryPreference.VEG,
        "vegetarian": DietaryPreference.VEG,
        "non-veg": DietaryPreference.NON_VEG,
        "nonveg": DietaryPreference.NON_VEG,
        "meat": DietaryPreference.NON_VEG,
        "vegan": DietaryPreference.VEGAN,
    }
    if key not in mapping:
        raise ValueError("Dietary preference must be Veg, Non-Veg, or Vegan (or 1/2/3).")
    return mapping[key]


def collect_user_profile(
    *,
    input_func: Callable[[str], str] | None = None,
) -> UserHealthProfile:
    """
    User Proxy Agent: gathers weight, height, age, gender, and dietary preference
    from the user and returns a validated profile.
    """
    read = input_func or input

    print("\n=== Step 1: User Proxy Agent ===")
    print("Enter your health profile (inputs are validated before the pipeline continues).\n")

    while True:
        try:
            weight_raw = read("Weight (kg): ")
            height_raw = read("Height (cm): ")
            age_raw = read("Age (years): ")
            gender_raw = read("Gender (male / female / other): ")
            print("Dietary preference: 1=Veg  2=Non-Veg  3=Vegan")
            diet_raw = read("Your choice: ")

            data = UserHealthProfile(
                weight_kg=_parse_float("Weight", weight_raw),
                height_cm=_parse_float("Height", height_raw),
                age=_parse_int("Age", age_raw),
                gender=_parse_gender(gender_raw),
                dietary_preference=_parse_diet(diet_raw),
            )
            logger.debug(
                "user_profile_collected",
                extra={
                    "action": "collect_user_profile",
                    "age": data.age,
                    "gender": data.gender.value,
                    "diet": data.dietary_preference.value,
                },
            )
            return data
        except (ValueError, ValidationError) as exc:
            logger.warning(
                "user_profile_validation_failed",
                extra={"action": "collect_user_profile", "error": str(exc)},
            )
            print(f"Could not validate inputs: {exc}\nPlease try again.\n")
