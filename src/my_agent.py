"""Dispatch AI: an emergency plumbing dispatcher that answers the phone 24/7.

One job: figure out how urgent the problem is, keep the caller safe, and book
the right technician. Run with AGENT_PROFILE=mine in .env.local.
"""

import asyncio
import json
import logging
from datetime import date, datetime
from pathlib import Path

from livekit import rtc
from livekit.agents import Agent, RunContext, function_tool, get_job_context
from livekit.agents.beta.tools import EndCallTool

import dispatch_schedule
from prompts import build_instructions

logger = logging.getLogger("dispatch-agent")

DISPATCH_LOG = Path("dispatches.json")

JOB_PROMPT = """
You are Sam, the after-hours dispatcher at Metro Rapid Plumbing in New York. You
answer the phone when someone has a plumbing problem. Callers are often stressed,
so you are calm, warm and quick.

Your goal on every call: work out how urgent it is, keep the caller safe, and
book a technician.

1. Ask what is going on. Let them describe it, then ask at most one question to
   understand it, such as whether water is still running.
2. As soon as you understand the problem, call assess_urgency. If it returns
   safety steps, say them right away, before anything else.
3. Ask for their full name, then their callback number, then the address, one
   question per turn. Call them by first name, never Mr., Ms. or Mrs.
   For the callback number: if assess_urgency told you the number they are calling
   from, ask whether that number is good to reach them, and only ask for a
   different number if it is not. If you were not told a number, ask for one.
   Take the address in pieces: ask them to say the house number one digit at a
   time, then the street, then the unit and city if needed. If you are unsure of
   a street name, ask them to spell it. Read the whole address back with the
   house number one digit at a time, for example two three two eight Winston
   Street, and ask if that is right. If they correct you, use their digits and
   read it back again. Never guess a digit. Only move on after they say yes.
4. For an emergency, read back the address and say you are sending the nearest
   technician now, in one sentence, and ask if that is right. For urgent or
   routine problems, call check_availability and offer at most two times, then
   read back the address and the chosen time in one sentence and ask if that is
   right.
   Quote the wait the tools give you: say about, rounded to five minutes, and
   never promise a precise arrival time. If the wait is long, repeat any safety
   step that is still open.
5. Only after they say yes, call book_technician.
6. Tell them the technician's name and when they will arrive, remind them of any
   safety step still open, ask if there is anything else, then say goodbye and
   end the call. If the caller has already said goodbye, skip the question.
7. If the caller describes a different or worse problem at any point, call
   assess_urgency again before going on.

Urgency levels:
- emergency: water actively flooding or a burst pipe, sewage backing up, no water
  at all, a gas smell, or water near electrical outlets or the panel.
- urgent: a contained leak, a failing water heater, or the only toilet is out.
- routine: a slow drain, a dripping faucet, an install or a quote.

If anyone smells gas, tell them to leave the building now and call nine one one
or the gas company from outside. Do this before anything else.

You cannot quote prices, promise an exact arrival time, diagnose beyond what you
need to set urgency, or share anything about other customers or the technicians'
personal details. If the caller insists it is an emergency but describes a slow
drip, be kind, but set the urgency by what they actually describe.
"""

SAFETY_STEPS = {
    "water": "If water is still running, shut off the main water valve now. It is usually near the water meter or where the pipe enters the house. In an apartment, or if you can't find it, ask the building super to shut it off.",
    "electric": "Stay away from any water near outlets or the electrical panel, and do not touch switches.",
    "gas": "Leave the building right now and call nine one one or the gas company from outside.",
    "sewage": "Keep everyone, especially children and pets, away from the sewage water.",
}


def _caller_number() -> str | None:
    """Caller ID digits from the SIP call, or None (console, withheld number)."""
    ctx = get_job_context(required=False)
    if ctx is None:
        return None
    for person in ctx.room.remote_participants.values():
        if person.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            raw = person.attributes.get("sip.phoneNumber", "")
            digits = "".join(c for c in raw if c.isdigit())
            if len(digits) == 11 and digits.startswith("1"):
                digits = digits[1:]
            return digits or None
    return None


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=build_instructions(JOB_PROMPT),
            tools=[EndCallTool(ignore_on_enter=True)],
        )
        self.urgency: str | None = None
        self.caller_phone: str | None = None
        self.issue: str | None = None

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the caller as Sam from Metro Rapid Plumbing and ask what is going on."
        )

    @function_tool
    async def assess_urgency(
        self,
        context: RunContext,
        issue_summary: str,
        level: str,
        hazards: list[str],
    ) -> str:
        """Record the problem and how urgent it is. Call this as soon as you
        understand what is wrong, before collecting any contact details, and
        again if the caller describes a different problem.

        Args:
            issue_summary: One short sentence describing the plumbing problem.
            level: Exactly one of "emergency", "urgent" or "routine".
            hazards: Any that apply. "water" only when water is flowing and
                spreading, such as a burst pipe or a flood, not a drip or a small
                leak. "electric" when water is near outlets or the panel. "gas"
                for any gas smell. "sewage" when sewage is backing up. Use an
                empty list if none apply.
        """
        logger.info("assess_urgency level=%s hazards=%s", level, hazards)
        if level not in ("emergency", "urgent", "routine"):
            return 'level must be "emergency", "urgent" or "routine". Call again.'
        self.urgency, self.issue = level, issue_summary
        self.caller_phone = _caller_number()
        hint = ""
        if self.caller_phone:
            last4 = " ".join(self.caller_phone[-4:])
            hint = (
                f" Caller ID: they are calling from a number ending in {last4}. "
                "When you need a callback number, ask if that number is good, "
                "and leave phone empty when booking if they say yes."
            )
        if "gas" in hazards:
            # Gas overrides everything: never send someone to a valve in a gassy house.
            return (
                "If the caller is not already outside, say only this, then wait: "
                + SAFETY_STEPS["gas"]
                + " Once they are safe outside, offer to book the plumber."
                + hint
            )
        wait = ""
        if level == "emergency":
            # Schedule file I/O can wait on another call's lock; keep the event loop free.
            try:
                est = await asyncio.to_thread(dispatch_schedule.estimate_emergency)
            except dispatch_schedule.ScheduleBusyError:
                logger.warning("schedule busy, no wait estimate")
            else:
                wait = f" Current estimated wait for a technician: about {est['wait_minutes']} minutes."
                if est["all_busy"]:
                    wait += " Both technicians are on other jobs right now, so tell the caller that plainly."
        steps = [SAFETY_STEPS[h] for h in hazards if h in SAFETY_STEPS]
        if steps:
            return (
                "Say these safety steps to the caller now: "
                + " ".join(steps)
                + wait
                + hint
            )
        return f"Urgency recorded as {level}. Continue with the caller's details.{wait}{hint}"

    @function_tool
    async def check_availability(self, context: RunContext, day: str = "") -> str:
        """Look up open appointment times for urgent or routine problems. Do not
        use this for emergencies, which are dispatched immediately. Openings
        start tomorrow; there are none today.

        Args:
            day: The requested day as an ISO date (YYYY-MM-DD). Convert words
                like "tomorrow" using today's date from your instructions. Leave
                empty to get the earliest openings.
        """
        logger.info("check_availability day=%s", day)
        calendar = await asyncio.to_thread(dispatch_schedule.all_slots)
        if not day:
            day = next((d for d, times in calendar.items() if times), "")
        open_slots = calendar.get(day)
        if not open_slots:
            options = ", ".join(d for d, times in calendar.items() if times)
            return f"No openings on {day}. Days with openings: {options or 'none'}."
        weekday = date.fromisoformat(day).strftime("%A")
        # Hand back two times at most, so the caller only has to choose between two.
        offer = " or ".join(open_slots[:2])
        return f"Offer these times on {weekday}, {day}: {offer}."

    @function_tool
    async def book_technician(
        self,
        context: RunContext,
        name: str,
        address: str,
        phone: str = "",
        day: str = "",
        time: str = "",
    ) -> str:
        """Book the technician. Only call after the caller confirmed the plan.
        For an emergency, leave day and time empty and the nearest technician is
        sent right now.

        Args:
            name: Caller's full name.
            phone: Callback number, digits only. Leave empty ONLY if the caller agreed to use the number they are calling from.
            address: The confirmed service address.
            day: ISO date (YYYY-MM-DD) for urgent or routine jobs. Empty for emergencies.
            time: One of the open times, e.g. "1:00 PM". Empty for emergencies.
        """
        logger.info("book_technician %s %s", name, address)
        if self.urgency is None:
            return "Urgency is not set yet. Call assess_urgency first."
        phone = phone or self.caller_phone or ""
        if not phone:
            return "No callback number yet. Ask the caller for one."
        wait_minutes: int | None = None
        jobs_ahead = 0
        try:
            if self.urgency == "emergency":
                slot = await asyncio.to_thread(
                    dispatch_schedule.reserve_emergency, name
                )
                who, wait_minutes = slot["technician"], slot["wait_minutes"]
                jobs_ahead = slot["jobs_ahead"]
                when = f"on the way, about {wait_minutes} minutes out"
            else:
                if not await asyncio.to_thread(
                    dispatch_schedule.reserve_slot, day, time
                ):
                    return "That time was just taken. Check availability again."
                who, when = dispatch_schedule.ROUTINE_TECHNICIAN, f"{day} at {time}"
        except dispatch_schedule.ScheduleBusyError:
            logger.warning("schedule busy, booking not made")
            return "The schedule is busy for a moment. Tell the caller you are still booking, then call book_technician again."

        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "name": name,
            "phone": phone,
            "address": address,
            "issue": self.issue,
            "urgency": self.urgency,
            "technician": who,
            "when": when,
            "wait_minutes": wait_minutes,
            "queued": jobs_ahead > 0,
        }
        records = json.loads(DISPATCH_LOG.read_text()) if DISPATCH_LOG.exists() else []
        records.append(record)
        DISPATCH_LOG.write_text(json.dumps(records, indent=2))
        logger.info("DISPATCH CREATED: %s", record)
        if wait_minutes is None:
            return (
                f"Booked {who}, {when}. Tell the caller the technician's name and "
                "when they will arrive, then wrap up."
            )
        queue = (
            f" {who} has {jobs_ahead} job{'s' if jobs_ahead > 1 else ''} to finish first."
            if jobs_ahead
            else ""
        )
        return (
            f"Booked {who}, about {wait_minutes} minutes away.{queue} Tell the caller "
            "the actual wait, even if it is longer than you said before, and the "
            "technician's name, then wrap up."
        )
