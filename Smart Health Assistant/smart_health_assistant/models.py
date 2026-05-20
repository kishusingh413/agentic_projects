from pydantic import BaseModel, Field

from smart_health_assistant.enums import DietaryPreference, Gender


class UserHealthProfile(BaseModel):
    """Validated inputs collected by the User Proxy Agent."""

    weight_kg: float = Field(gt=0, le=500, description="Weight in kilograms")
    height_cm: float = Field(gt=0, le=300, description="Height in centimeters")
    age: int = Field(ge=1, le=120, description="Age in years")
    gender: Gender
    dietary_preference: DietaryPreference
