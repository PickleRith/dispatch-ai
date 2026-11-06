"""Entry point: wires the voice pipeline and picks which agent answers.

You mostly edit my_agent.py. Come here to change models, silence handling or
turn-taking.

  .env.local
    AGENT_PROFILE = booking | screener | mine     (which agent answers)
    MODEL_PRESET  = budget  | quality             (see models.py)
    AGENT_NAME    = take2-agent                   (must match your dispatch rule)
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatMessage,
    ConversationItemAddedEvent,
    JobContext,
    STTContextOptions,
    TurnHandlingOptions,
    UserStateChangedEvent,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics

from booking_agent import BookingAgent
from models import load_models
from my_agent import MyAgent
from screener_agent import ScreenerAgent

logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENTS: dict[str, type[Agent]] = {
    "booking": BookingAgent,
    "screener": ScreenerAgent,
    "mine": MyAgent,
}

AGENT_NAME = os.getenv("AGENT_NAME", "take2-agent")

# Silence: after this many quiet seconds the caller counts as "away".
SILENCE_SECONDS = 8.0
# Check in this many times, then say goodbye and hang up.
MAX_CHECK_INS = 2


def pick_agent() -> Agent:
    profile = os.getenv("AGENT_PROFILE", "booking").strip().lower()
    if profile not in AGENTS:
        raise ValueError(
            f"Unknown AGENT_PROFILE '{profile}'. Choose one of: {', '.join(AGENTS)}"
        )
    return AGENTS[profile]()


def handle_silence(session: AgentSession) -> None:
    """Ask "are you still there?" when the caller goes quiet, then hang up."""
    check_ins = 0
    tasks: set[asyncio.Task[None]] = set()

    async def check_in() -> None:
        nonlocal check_ins
        check_ins += 1
        if check_ins <= MAX_CHECK_INS:
            await session.generate_reply(
                instructions="The caller has gone quiet. In a few words, ask if they are still there."
            )
        else:
            await session.generate_reply(
                instructions="The caller is not responding. Say a short, polite goodbye."
            )
            session.shutdown()

    @session.on("user_state_changed")
    def _on_user_state(ev: UserStateChangedEvent) -> None:
        nonlocal check_ins
        if ev.new_state == "speaking":
            check_ins = 0
        elif ev.new_state == "away":
            task = asyncio.create_task(check_in())
            tasks.add(task)
            task.add_done_callback(tasks.discard)


def log_latency(session: AgentSession) -> None:
    """Log how long each reply took, so model and setting changes can be compared."""

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if not isinstance(item, ChatMessage):
            return
        m = item.metrics
        if item.role == "user" and "end_of_turn_delay" in m:
            logger.info("latency user: end_of_turn=%.2fs", m["end_of_turn_delay"])
        elif item.role == "assistant" and "e2e_latency" in m:
            logger.info(
                "latency reply: total=%.2fs llm_ttft=%.2fs tts_ttfb=%.2fs",
                m["e2e_latency"],
                m.get("llm_node_ttft", -1.0),
                m.get("tts_node_ttfb", -1.0),
            )


# Keep one process loaded and waiting, so the first call after a restart is
# answered right away. Dev mode defaults to 0, which made callers say "hello?".
server = AgentServer(num_idle_processes=1)


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    models = load_models()

    session = AgentSession(
        # Ears: speech-to-text
        stt=models.stt,
        # Bias transcription toward words it would otherwise misspell.
        stt_context_options=STTContextOptions(
            keyterms=[
                "Metro Rapid Plumbing",
                "Astoria",
                "Brooklyn",
                "Queens",
                "Bronx",
                "Manhattan",
                "Staten Island",
                "Jersey City",
                "shut-off valve",
                "water heater",
            ],
            keyterm_detection={"enabled": True},
        ),
        # Brain: the LLM
        llm=models.llm,
        # Voice: text-to-speech
        tts=models.tts,
        expressive=models.expressive,
        turn_handling=TurnHandlingOptions(
            # Decides when the caller has finished speaking, using both the words
            # and the audio (pitch, pauses). Much better than a fixed silence timer.
            turn_detection=inference.TurnDetector(),
            # How long to wait after the caller stops before replying.
            # min 0.6s: the default 0.3s cut callers off mid-thought on test calls.
            # max 1.8s: used when the turn detector is unsure; 3.0s made those
            # replies take ~3.4s. Dynamic mode learns each caller's pace in between.
            endpointing={"mode": "dynamic", "min_delay": 0.6, "max_delay": 1.8},
            # Start drafting a reply before the turn is confirmed. Cuts latency.
            preemptive_generation={"enabled": True},
        ),
        user_away_timeout=SILENCE_SECONDS,
    )

    handle_silence(session)
    log_latency(session)

    async def log_usage() -> None:
        # Shows model usage per call so you can see what you're spending.
        logger.info("session usage: %s", session.usage)

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=pick_agent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Removes background noise and other voices before STT hears it.
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
            # Hang up the phone line when the session ends.
            delete_room_on_close=True,
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
