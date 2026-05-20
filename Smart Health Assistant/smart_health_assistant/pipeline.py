import asyncio
import json
import logging
from dataclasses import dataclass

from smart_health_assistant.config import Settings
from smart_health_assistant.llm_agents import (
    create_bmi_agent,
    create_diet_planner_agent,
    create_model_client,
    create_workout_scheduler_agent,
)
from smart_health_assistant.message_utils import final_agent_text
from smart_health_assistant.models import UserHealthProfile
from smart_health_assistant.services.bmi_service import bmi_category, format_bmi_insights
from smart_health_assistant.services.diet_service import build_meal_plan
from smart_health_assistant.services.workout_service import build_weekly_workout
from smart_health_assistant.tools.bmi_tool import calculate_bmi
from smart_health_assistant.user_proxy import collect_user_profile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthPlanResult:
    profile: UserHealthProfile
    bmi_section: str
    diet_section: str
    workout_section: str
    mode: str

    def as_report(self) -> str:
        return (
            "\n\n"
            + "=" * 60
            + "\n SMART HEALTH PLAN (complete)\n"
            + "=" * 60
            + f"\n\n## BMI insights\n{self.bmi_section}\n\n"
            + f"## Diet plan\n{self.diet_section}\n\n"
            + f"## Weekly workouts\n{self.workout_section}\n"
        )


async def _run_llm_pipeline(profile: UserHealthProfile, settings: Settings) -> HealthPlanResult:
    client = create_model_client(model=settings.openai_model, api_key=settings.openai_api_key or "")
    try:
        bmi_agent = create_bmi_agent(client)
        diet_agent = create_diet_planner_agent(client)
        workout_agent = create_workout_scheduler_agent(client)

        bmi_task = (
            f"User metrics: height_cm={profile.height_cm}, weight_kg={profile.weight_kg}. "
            "Call calculate_bmi with these values, then respond with analysis and recommendations."
        )
        print("\n>>> Step 2-3: BMI Agent (registered BMI tool: calculate_bmi) ...")
        try:
            bmi_result = await bmi_agent.run(task=bmi_task)
        except Exception as exc:
            logger.exception(
                "bmi_agent_failed",
                extra={
                    "action": "run_llm_pipeline",
                    "stage": "bmi_agent",
                    "height_cm": profile.height_cm,
                    "weight_kg": profile.weight_kg,
                },
            )
            raise RuntimeError("BMI Agent step failed.") from exc
        bmi_text = final_agent_text(bmi_result)
        if not bmi_text.strip():
            raise RuntimeError("BMI Agent returned no text output.")

        diet_task = (
            "BMI insights from the BMI Agent:\n"
            f"{bmi_text}\n\n"
            f"Dietary preference: {profile.dietary_preference.value}. "
            "Produce the meal plan as instructed in your system message."
        )
        print("\n>>> Step 4: Diet Planner Agent ...")
        try:
            diet_result = await diet_agent.run(task=diet_task)
        except Exception as exc:
            logger.exception(
                "diet_agent_failed",
                extra={"action": "run_llm_pipeline", "stage": "diet_planner_agent"},
            )
            raise RuntimeError("Diet Planner Agent step failed.") from exc
        diet_text = final_agent_text(diet_result)
        if not diet_text.strip():
            raise RuntimeError("Diet Planner Agent returned no text output.")

        workout_task = (
            "Meal plan from the Diet Planner Agent:\n"
            f"{diet_text}\n\n"
            f"User age: {profile.age}. User gender: {profile.gender.value}. "
            "Produce the 7-day schedule as instructed in your system message."
        )
        print("\n>>> Step 5: Workout Scheduler Agent ...")
        try:
            workout_result = await workout_agent.run(task=workout_task)
        except Exception as exc:
            logger.exception(
                "workout_agent_failed",
                extra={"action": "run_llm_pipeline", "stage": "workout_scheduler_agent"},
            )
            raise RuntimeError("Workout Scheduler Agent step failed.") from exc
        workout_text = final_agent_text(workout_result)
        if not workout_text.strip():
            raise RuntimeError("Workout Scheduler Agent returned no text output.")

        return HealthPlanResult(
            profile=profile,
            bmi_section=bmi_text.strip(),
            diet_section=diet_text.strip(),
            workout_section=workout_text.strip(),
            mode="llm",
        )
    finally:
        await client.close()


def _run_offline_pipeline(profile: UserHealthProfile) -> HealthPlanResult:
    """Deterministic pipeline mirroring the same agent order when no LLM is configured."""
    print("\n>>> Step 2: BMI tool (calculate_bmi) ...")
    tool_json = calculate_bmi(profile.height_cm, profile.weight_kg)
    try:
        payload = json.loads(tool_json)
    except json.JSONDecodeError as exc:
        logger.exception(
            "bmi_tool_json_invalid",
            extra={"action": "offline_pipeline", "raw": tool_json},
        )
        raise RuntimeError("BMI tool returned invalid JSON.") from exc

    bmi_value = float(payload.get("bmi", 0))
    category = str(payload.get("category", ""))
    if not category:
        category = bmi_category(bmi_value)

    print(">>> Step 3: BMI Agent insights (rule-based) ...")
    bmi_text = format_bmi_insights(bmi_value, category)
    print(">>> Step 4: Diet Planner Agent (rule-based) ...")
    diet_text = build_meal_plan(profile, category)
    print(">>> Step 5: Workout Scheduler Agent (rule-based) ...")
    workout_text = build_weekly_workout(profile, diet_text)

    return HealthPlanResult(
        profile=profile,
        bmi_section=bmi_text,
        diet_section=diet_text,
        workout_section=workout_text,
        mode="offline",
    )


async def run_health_assistant(settings: Settings) -> HealthPlanResult:
    profile = collect_user_profile()
    if settings.llm_enabled():
        logger.debug("pipeline_mode", extra={"action": "run_health_assistant", "mode": "llm"})
        return await _run_llm_pipeline(profile, settings)
    logger.warning(
        "pipeline_mode_offline",
        extra={
            "action": "run_health_assistant",
            "mode": "offline",
            "reason": "OPENAI_API_KEY missing or empty — using built-in recommendations.",
        },
    )
    return _run_offline_pipeline(profile)


def run_health_assistant_sync(settings: Settings) -> HealthPlanResult:
    return asyncio.run(run_health_assistant(settings))
