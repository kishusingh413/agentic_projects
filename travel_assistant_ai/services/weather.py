"""WeatherAPI.com client — business logic for current conditions."""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

WEATHER_CURRENT_URL = "https://api.weatherapi.com/v1/current.json"
REQUEST_TIMEOUT_SECONDS = 30


def fetch_current_weather(destination: str, api_key: str) -> str:
    """
    Fetch current weather for *destination* and return a short text summary.

    *api_key* is the WeatherAPI.com key (validated at app startup).
    """
    place = destination.strip()
    if not place:
        return "Error: no destination was provided for the weather lookup."

    params: dict[str, str | int] = {"key": api_key, "q": place, "aqi": "no"}

    try:
        response = requests.get(
            WEATHER_CURRENT_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception(
            "Weather API request failed",
            extra={"action": "fetch_current_weather", "destination": place},
        )
        return f"Weather service request failed: {exc}"

    try:
        data: dict[str, Any] = response.json()
    except ValueError:
        logger.error(
            "Weather API returned non-JSON body",
            extra={"action": "fetch_current_weather", "destination": place},
        )
        return "Weather service returned invalid data."

    error_payload = data.get("error")
    if isinstance(error_payload, dict):
        message = error_payload.get("message")
        if isinstance(message, str):
            return f"Weather API error: {message}"
        return "Weather API returned an error."

    location = data.get("location")
    current = data.get("current")
    if not isinstance(location, dict) or not isinstance(current, dict):
        return (
            "Weather response was missing expected fields; try another spelling "
            "or a nearby major city."
        )

    name = location.get("name")
    region = location.get("region")
    country = location.get("country")
    loc_bits = [b for b in (name, region, country) if isinstance(b, str) and b]
    loc_line = ", ".join(loc_bits) if loc_bits else place

    temp_c = current.get("temp_c")
    condition_block = current.get("condition")
    condition_text = None
    if isinstance(condition_block, dict):
        raw = condition_block.get("text")
        condition_text = raw if isinstance(raw, str) else None
    humidity = current.get("humidity")
    wind_kph = current.get("wind_kph")

    lines: list[str] = [f"Resolved location: {loc_line}"]
    if temp_c is not None:
        lines.append(f"Temperature: {temp_c} °C")
    if condition_text is not None:
        lines.append(f"Condition: {condition_text}")
    if humidity is not None:
        lines.append(f"Humidity: {humidity}%")
    if wind_kph is not None:
        lines.append(f"Wind: {wind_kph} km/h")

    return "\n".join(lines)
