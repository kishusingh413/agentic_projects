from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from smart_health_assistant.tools.bmi_tool import calculate_bmi


def create_model_client(*, model: str, api_key: str) -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
    )


def create_bmi_agent(client: OpenAIChatCompletionClient) -> AssistantAgent:
    """BMI Agent with the BMI calculation tool registered (tools=[calculate_bmi])."""
    return AssistantAgent(
        name="bmi_agent",
        model_client=client,
        tools=[calculate_bmi],
        reflect_on_tool_use=True,
        max_tool_iterations=4,
        description="Computes BMI via the registered tool and interprets results for the user.",
        system_message=(
            "You are the BMI Agent in a Smart Health Assistant. "
            "When the user message includes height_cm and weight_kg, you MUST call the tool "
            "`calculate_bmi` with those exact numeric values (height in centimeters, weight in kilograms). "
            "After the tool returns JSON, explain the BMI value and category in clear language and provide "
            "practical health and lifestyle recommendations. "
            "If the user has medical conditions, remind them to consult a clinician. "
            "Do not guess BMI without calling the tool."
        ),
    )


def create_diet_planner_agent(client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="diet_planner_agent",
        model_client=client,
        reflect_on_tool_use=False,
        description="Builds meal suggestions from BMI insights and dietary preference.",
        system_message=(
            "You are the Diet Planner Agent. "
            "You receive BMI insights from the BMI Agent and the user's dietary preference "
            "(Veg, Non-Veg, or Vegan). "
            "Produce a tailored one-day meal plan: breakfast, lunch, dinner, and two snacks, "
            "with simple portion cues. Align energy density with the BMI category when appropriate. "
            "Keep guidance general wellness—not a prescription—and note that a dietitian can personalize further."
        ),
    )


def create_workout_scheduler_agent(client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="workout_scheduler_agent",
        model_client=client,
        reflect_on_tool_use=False,
        description="Schedules weekly workouts using age, gender, and meal-plan context.",
        system_message=(
            "You are the Workout Scheduler Agent. "
            "You receive meal-plan suggestions from the Diet Planner Agent plus the user's age and gender. "
            "Create a balanced 7-day weekly workout plan (e.g., strength, cardio, mobility, rest). "
            "Account for age with sensible intensity and recovery. "
            "Reference how fueling aligns with the meal plan where helpful. "
            "Avoid medical claims; encourage professional guidance for injuries or chronic conditions."
        ),
    )
