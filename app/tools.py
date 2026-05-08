"""Mock tools the agent can call.

Currently a single tool: ``check_available_slots(department, date)``.
Returns hard-coded synthetic availability so the demo works without any
external integration. The schema mirrors what a real implementation
would return so it would be a drop-in replacement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from app.logger import get_logger, log_event

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Synthetic availability schedule

# A small table the panel can inspect easily. In a real system this would be
# a query against a scheduling backend.
_DEPARTMENT_SCHEDULES: dict[str, dict] = {
    "general medicine": {
        "weekdays_open": {0, 1, 2, 3, 4},  # Mon-Fri
        "morning_slots": ["09:00", "09:30", "10:00", "10:30", "11:00"],
        "afternoon_slots": ["14:00", "14:30", "15:30", "16:00"],
    },
    "paediatrics": {
        "weekdays_open": {0, 1, 2, 3, 4},
        "morning_slots": ["09:00", "10:00", "11:00"],
        "afternoon_slots": ["14:00", "15:00", "16:00"],
    },
    "dermatology": {
        "weekdays_open": {1, 3},  # Tue, Thu only
        "morning_slots": ["10:00", "11:00"],
        "afternoon_slots": ["15:00", "16:00"],
    },
    "cardiology": {
        "weekdays_open": {0, 2, 4},  # Mon, Wed, Fri
        "morning_slots": ["09:30", "10:30", "11:30"],
        "afternoon_slots": [],
    },
    "nutrition": {
        "weekdays_open": {1, 2, 3, 4},
        "morning_slots": ["10:00"],
        "afternoon_slots": ["14:00", "15:00"],
    },
}

_DEPARTMENT_ALIASES = {
    "gp": "general medicine",
    "general practice": "general medicine",
    "general": "general medicine",
    "pediatrics": "paediatrics",
    "kids": "paediatrics",
    "children": "paediatrics",
    "skin": "dermatology",
    "heart": "cardiology",
    "cardiac": "cardiology",
    "diet": "nutrition",
    "dietician": "nutrition",
    "dietitian": "nutrition",
}


@dataclass
class SlotAvailability:
    department: str
    date: str  # ISO yyyy-mm-dd
    available: bool
    reason: str | None
    morning_slots: list[str]
    afternoon_slots: list[str]

    def as_dict(self) -> dict:
        return {
            "department": self.department,
            "date": self.date,
            "available": self.available,
            "reason": self.reason,
            "morning_slots": self.morning_slots,
            "afternoon_slots": self.afternoon_slots,
        }


# ---------------------------------------------------------------------------
# Argument extraction


_KNOWN_DEPARTMENTS = list(_DEPARTMENT_SCHEDULES.keys())

_DAY_OF_WEEK = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _normalise_department(name: str | None, fallback_text: str) -> str | None:
    """Find a known department in either ``name`` or the broader text."""
    haystacks = []
    if name:
        haystacks.append(name.lower())
    haystacks.append(fallback_text.lower())

    for hay in haystacks:
        for dept in _KNOWN_DEPARTMENTS:
            if dept in hay:
                return dept
        for alias, target in _DEPARTMENT_ALIASES.items():
            if alias in hay:
                return target
    return None


def _parse_date(text: str, today: date | None = None) -> date | None:
    """Extract a date from free-form text. Returns None if nothing found."""
    today = today or date.today()
    lowered = text.lower()

    if "today" in lowered:
        return today
    if "tomorrow" in lowered:
        return today + timedelta(days=1)

    # Day-of-week: pick the next occurrence (including today if it matches).
    for name, weekday in _DAY_OF_WEEK.items():
        if name in lowered:
            offset = (weekday - today.weekday()) % 7
            offset = offset if offset != 0 else 7  # "Monday" said on Monday => next Mon
            return today + timedelta(days=offset)

    # ISO date yyyy-mm-dd
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_match:
        try:
            return datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    # dd/mm or dd-mm-yyyy
    dmy_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", text)
    if dmy_match:
        day, month, year = dmy_match.groups()
        year = year or str(today.year)
        if len(year) == 2:
            year = "20" + year
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            return None

    return None


def extract_arguments(question: str) -> tuple[str | None, date | None]:
    """Return (department, date) extracted from a free-text question."""
    return _normalise_department(None, question), _parse_date(question)


# ---------------------------------------------------------------------------
# The tool itself


def check_available_slots(
    department: str | None,
    date_value: date | str | None,
) -> SlotAvailability:
    """Return mock availability for a given department + date."""

    # Normalise inputs.
    normalised_dept = _normalise_department(department, department or "")
    if isinstance(date_value, str):
        date_obj = _parse_date(date_value)
    else:
        date_obj = date_value
    if date_obj is None:
        date_obj = date.today() + timedelta(days=1)

    iso_date = date_obj.isoformat()

    if normalised_dept is None:
        return SlotAvailability(
            department=department or "(unspecified)",
            date=iso_date,
            available=False,
            reason=(
                "I could not identify the department. Please specify one of: "
                + ", ".join(_KNOWN_DEPARTMENTS) + "."
            ),
            morning_slots=[],
            afternoon_slots=[],
        )

    schedule = _DEPARTMENT_SCHEDULES[normalised_dept]
    weekday = date_obj.weekday()

    if weekday not in schedule["weekdays_open"]:
        result = SlotAvailability(
            department=normalised_dept,
            date=iso_date,
            available=False,
            reason=(
                f"{normalised_dept.title()} does not have clinics on "
                f"{date_obj.strftime('%A')}s."
            ),
            morning_slots=[],
            afternoon_slots=[],
        )
    else:
        result = SlotAvailability(
            department=normalised_dept,
            date=iso_date,
            available=bool(schedule["morning_slots"] or schedule["afternoon_slots"]),
            reason=None,
            morning_slots=list(schedule["morning_slots"]),
            afternoon_slots=list(schedule["afternoon_slots"]),
        )

    log_event(
        log,
        "tool.check_available_slots",
        department=result.department,
        date=result.date,
        available=result.available,
    )
    return result


def format_slot_response(slots: SlotAvailability) -> str:
    """Render a SlotAvailability into a friendly natural-language answer."""
    if not slots.available:
        return (
            f"I checked mock appointment availability for {slots.department} "
            f"on {slots.date}: {slots.reason or 'no slots are available.'}"
        )

    morning = ", ".join(slots.morning_slots) or "none"
    afternoon = ", ".join(slots.afternoon_slots) or "none"
    return (
        f"I checked mock appointment availability for {slots.department.title()} "
        f"on {slots.date}.\n"
        f"- Morning slots: {morning}\n"
        f"- Afternoon slots: {afternoon}\n\n"
        "Note: this is mock data from the demo scheduler, not a real booking system."
    )
