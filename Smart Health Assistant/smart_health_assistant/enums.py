from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class DietaryPreference(str, Enum):
    VEG = "veg"
    NON_VEG = "non-veg"
    VEGAN = "vegan"
