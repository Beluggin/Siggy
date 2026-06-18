# signalbot.py
"""
═══════════════════════════════════════════════════════════════════
SIGNALBOT v6.3 — Cognitive Mode Registry + Remember Mode
═══════════════════════════════════════════════════════════════════

FIX LOG:
  v6.0: Daemon ran empty, LLM didn't understand temporal experience
  v6.1: Boot seeding + prompt framing (daemon thinks, LLM knows)
  v6.2: Goal lifecycle — conversation topics feed into daemon,
        resolved topics retire, boot goals expire.
  v6.3: Cognitive Mode Registry — modes as immune response to
        cognitive insufficiency. Remember Mode for deep archive
        retrieval. Gap detection for future mode discovery.
        Memory archival compresses old turns into episodes.

COMMANDS:
  state     — cognitive state vectors + active modes
  facts     — learned indelible facts
  daemon    — daemon status + what it's been thinking
  curiosity — curiosity signal breakdown
  modes     — cognitive mode status + recent gaps
  archive   — archive stats + force archive
  dream on/off — toggle dream mode
  exit/quit — shutdown
  read code - show Signalbot the code
  search - run a duckduckgo search
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import signal_ethics

from persistent_behavior import PersistentBehaviorModifier
from paradox_protection import ParadoxProtector
from response_engine import generate_response
import response_engine
import telemetry_log
from memory_engine import save_interaction, load_recent_memory

from cognitive_state import get_cognitive_state, get_tone_instructions
from indelible_facts import register_fact, get_indelible_prompt_section
from memory_twdc_stateful import load_long_memory_block_stateful, get_stateful_twdc
from plan_buffer import get_plan_buffer
from temporal_daemon import get_daemon
from goal_engine_DAEMON import GoalEngine as DaemonGoalEngine
from curiosity_engine import get_curiosity_signal, get_curiosity_report

from cognitive_modes import get_mode_engine
from memory_archive import archive_old_memories, get_archive_stats, force_archive_all
from code_reader import get_code_context, get_file_context, get_file_list_brief
from temporal_integrity_UPDATED import get_temporal_integrity
from speech_interface_v2 import get_speech_interface, speak, listen_push_to_talk
from web_search import web_search, news_search, format_search_for_prompt, format_news_for_prompt, print_search_results, print_news_results
try:
    import evolve_adapter  # Non-critical: no attributes called; exists for evolve loop compatibility
    EVOLVE_ADAPTER_AVAILABLE = True
except ImportError:
    EVOLVE_ADAPTER_AVAILABLE = False

try:
    from intent_codelet.intent_codelet import classify_intent
    INTENT_AVAILABLE = True
except ImportError:
    INTENT_AVAILABLE = False

ETHOS_CHECKSUM = "7197c555c2c6ddc845a410529f021e0d511ad6951d80abee09b976a34867384e"
INTENT_BYPASS = not INTENT_AVAILABLE

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"[TIME] {label}: {dt_ms:9.1f} ms")

def clamp_torch_threads():
    try:
        import torch
        torch.set_num_threads(1)
        print(f"[Torch] num_threads={torch.get_num_threads()}")
    except Exception as e:
        print(f"[Torch] not available: {e}")

def verify_ethos_integrity() -> bool:
    with open("signal_ethics.py", "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest() == ETHOS_CHECKSUM

def load_identity_prompt(path: str = "signal_identity.txt") -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return "You are SignalBot. Clever, candid, and slightly irreverent."

class _IntentStub:
    def __init__(self, label="GENERAL", confidence=1.0):
        self.label = label
        self.confidence = confidence


def main():
    # ═══ INITIALIZATION ═══
    clamp_torch_threads()

    if not verify_ethos_integrity():
        raise SystemExit("Ethos integrity compromised.")

    personality_prompt = load_identity_prompt()
    behavior_mod = PersistentBehaviorModifier()
    paradox_guard = ParadoxProtector()

    mem_stateful = get_stateful_twdc()
    cog_state = get_cognitive_state()
    temp_integrity = get_temporal_integrity()

    daemon_goals = DaemonGoalEngine()
    daemon = get_daemon(goal_engine=daemon_goals)
    mode_engine = get_mode_engine()
    buf = get_plan_buffer()

    print("[INIT] State-aware memory engine initialized")
    print("[INIT] Daemon goal engine initialized")
    print(f"[INIT] Cognitive mode engine initialized ({len(mode_engine._modes)} modes)")

    # Run archival on startup (compress old memories)
    archived = archive_old_memories()
    if archived > 0:
        mode_engine.refresh_archive_tags()
        print(f"[INIT] Archived {archived} episodes from old memory")
    speech = get_speech_interface()
    print("[INIT] Speech interface initialized")

    # Model selection
    print("\nSelect model:")
    print("1.  Claude Opus 4.7 (API, smartest)")
    print("2.  Claude Sonnet 4.6 (API, smart + fast)")
    print("3.  Claude Haiku 4.5 (API, fastest)")
    print("4.  Mistral Medium (API)")
    print("5.  Mistral 7b (local)")
    print("6.  Mistral 7b Q8 (local, higher quality)")
    print("7.  Llama 3.1 8b Q8 (local)")
    print("8.  Gemma2:9b (local)")
    print("9.  Phi3 (local)")
    print("10. Phi4 Full (local)")
    print("11. Gemma4:12b (local)")
    print("12. Gemini 3.1 Pro (API)")
    print("13. Gemini 3.1 Flash (API)")
    print("14. Gemini Deep Research (API, Apr 2026)")
    print("15. DeepSeek R1 (API, needs DEEPSEEK_API_KEY)")
    print("16. GPT-4.1 (API, needs OPENAI_KEY)")
    print("17. GPT-4o (API, needs OPENAI_KEY)")
    print("18. ChatGPT 5.5 (API, needs OPENAI_KEY)")
    print("19. Claude Fable 5 (API, claude-fable-5)")
    while True:
        choice = input("Enter 1-19: ").strip()
        if choice == "1":
            response_engine.USE_ANTHROPIC = True
            response_engine.USE_MISTRAL = False
            response_engine.ANTHROPIC_MODEL = "claude-opus-4-7"
            print("✓ Using Claude Opus 4.7\n")
            break
        elif choice == "2":
            response_engine.USE_ANTHROPIC = True
            response_engine.USE_MISTRAL = False
            response_engine.ANTHROPIC_MODEL = "claude-sonnet-4-6"
            print("✓ Using Claude Sonnet 4.6\n")
            break
        elif choice == "3":
            response_engine.USE_ANTHROPIC = True
            response_engine.USE_MISTRAL = False
            response_engine.ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
            print("✓ Using Claude Haiku 4.5\n")
            break
        elif choice == "4":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = True
            print("✓ Using Mistral Medium\n")
            break
        elif choice == "5":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.OLLAMA_MODEL = "mistral:7b"
            print("✓ Using Mistral 7b\n")
            break
        elif choice == "6":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.OLLAMA_MODEL = "mistral:7b-instruct-q8_0"
            print("✓ Using Mistral 7b Q8\n")
            break
        elif choice == "7":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.OLLAMA_MODEL = "llama3.1:8b-instruct-q8_0"
            print("✓ Using Llama 3.1 8b Q8\n")
            break
        elif choice == "8":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.OLLAMA_MODEL = "gemma2:9b"
            print("✓ Using Gemma2:9b\n")
            break
        elif choice == "9":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.OLLAMA_MODEL = "phi3:latest"
            print("✓ Using Phi3\n")
            break
        elif choice == "10":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.OLLAMA_MODEL = "phi4:latest"
            print("✓ Using Phi4 Full\n")
            break
        elif choice == "11":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.OLLAMA_MODEL = "gemma4:12b"
            print("✓ Using Gemma4:12b\n")
            break
        elif choice == "12":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.USE_GEMINI = True
            response_engine.USE_DEEPSEEK = False
            response_engine.GEMINI_MODEL = "gemini-2.5-pro"
            print("✓ Using Gemini 2.5 Pro\n")
            break
        elif choice == "13":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.USE_GEMINI = True
            response_engine.USE_DEEPSEEK = False
            response_engine.GEMINI_MODEL = "gemini-2.5-flash"
            print("✓ Using Gemini 2.5 Flash\n")
            break
        elif choice == "14":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.USE_GEMINI = True
            response_engine.USE_DEEPSEEK = False
            response_engine.GEMINI_MODEL = "gemini-3.5-flash"
            print("✓ Using Gemini 3.5 Flash\n")
            break
        elif choice == "15":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.USE_GEMINI = False
            response_engine.USE_DEEPSEEK = True
            response_engine.USE_OPENAI = False
            response_engine.DEEPSEEK_MODEL = "deepseek-reasoner"
            print("✓ Using DeepSeek R1\n")
            break
        elif choice == "16":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.USE_GEMINI = False
            response_engine.USE_DEEPSEEK = False
            response_engine.USE_OPENAI = True
            response_engine.OPENAI_MODEL = "gpt-4.1"
            print("✓ Using GPT-4.1\n")
            break
        elif choice == "17":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.USE_GEMINI = False
            response_engine.USE_DEEPSEEK = False
            response_engine.USE_OPENAI = True
            response_engine.OPENAI_MODEL = "gpt-4o"
            print("✓ Using GPT-4o\n")
            break
        elif choice == "18":
            response_engine.USE_ANTHROPIC = False
            response_engine.USE_MISTRAL = False
            response_engine.USE_GEMINI = False
            response_engine.USE_DEEPSEEK = False
            response_engine.USE_OPENAI = True
            response_engine.OPENAI_MODEL = "gpt-5.5"
            print("✓ Using ChatGPT 5.5\n")
            break
        elif choice == "19":
            response_engine.USE_ANTHROPIC = True
            response_engine.USE_MISTRAL = False
            response_engine.USE_GEMINI = False
            response_engine.USE_DEEPSEEK = False
            response_engine.USE_OPENAI = False
            # Fable = Claude tier above Opus; launched 2026-06-09
            response_engine.ANTHROPIC_MODEL = "claude-fable-5"
            print("✓ Using Claude Fable 5\n")
            break
        else:
            print("Invalid choice, try again.")

    # ═══ START DAEMON ═══
    daemon.start()

    print("🟢 SignalBot v6.3 Online (Cognitive Modes + Remember Mode)")
    print("Commands: 'state', 'facts', 'daemon', 'curiosity', 'modes', 'archive', 'plans', 'dream on/off', 'speech on/off', 'model', 'model <1-19>', 'exit', 'read code'")

    # ═══ MAIN LOOP ═══
    dream_mode = True
    speech_mode = False    
    turn = 0
    last_bot_output = ""
    code_injection = None
    search_injection = None

    try:
        while True:
            if speech_mode:
                print("You: ", end="", flush=True)
                user_input = listen_push_to_talk(timeout=10.0)  # 10 second timeout
                if user_input is None:
                    print("(no speech detected, type instead)")
                    user_input = input().strip()
                else:
                    print(user_input)  # Show what was recognized
            else:
                user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break

            # ─── COMMAND HANDLING ───
            if user_input.lower() == "dream on":
                dream_mode = True
                print("[MODE] Dream mode ON\n")
                continue
            if user_input.lower() == "dream off":
                dream_mode = False
                print("[MODE] Dream mode OFF\n")
                continue
            if user_input.lower() == "speech on":
                if speech.enable():
                    speech_mode = True
                    speak("Speech mode enabled. I can hear and speak now.")
                else:
                    print("[ERROR] Speech system not available\n")
                continue

            if user_input.lower() == "speech off":
                speech.disable()
                speech_mode = False
                print("[MODE] Speech mode OFF\n")
                continue

            if user_input.lower() == "speech status":
                print(f"\n{speech.get_status()}\n")
                continue
            
            if user_input.lower() == "plans":
                print(f"\n{buf.get_full_report()}\n")
                continue

            if user_input.lower().startswith("model"):
                # (num_str, model_id, use_anthropic, use_mistral, use_gemini, use_deepseek, use_openai)
                _MODEL_CHOICES = [
                    ("1",  "claude-opus-4-7",            True,  False, False, False, False),
                    ("2",  "claude-sonnet-4-6",          True,  False, False, False, False),
                    ("3",  "claude-haiku-4-5-20251001",  True,  False, False, False, False),
                    ("4",  None,                         False, True,  False, False, False),  # Mistral Medium
                    ("5",  "mistral:7b",                 False, False, False, False, False),
                    ("6",  "mistral:7b-instruct-q8_0",   False, False, False, False, False),
                    ("7",  "llama3.1:8b-instruct-q8_0", False, False, False, False, False),
                    ("8",  "gemma2:9b",                  False, False, False, False, False),
                    ("9",  "phi3:latest",                False, False, False, False, False),
                    ("10", "phi4:latest",                False, False, False, False, False),
                    ("11", "gemma4:12b",                 False, False, False, False, False),
                    ("12", "gemini-2.5-pro",              False, False, True,  False, False),
                    ("13", "gemini-2.5-flash",            False, False, True,  False, False),
                    ("14", "gemini-3.5-flash",            False, False, True,  False, False),
                    ("15", "deepseek-reasoner",           False, False, False, True,  False),
                    ("16", "gpt-4.1",                     False, False, False, False, True),
                    ("17", "gpt-4o",                      False, False, False, False, True),
                    ("18", "gpt-5.5",                     False, False, False, False, True),
                    ("19", "claude-fable-5",              True,  False, False, False, False),  # Fable 5
                ]
                _MODEL_LABELS = [
                    "Claude Opus 4.7 (API)",
                    "Claude Sonnet 4.6 (API)",
                    "Claude Haiku 4.5 (API)",
                    "Mistral Medium (API)",
                    "Mistral 7b (local)",
                    "Mistral 7b Q8 (local)",
                    "Llama 3.1 8b Q8 (local)",
                    "Gemma2:9b (local)",
                    "Phi3 (local)",
                    "Phi4 Full (local)",
                    "Gemma4:12b (local)",
                    "Gemini 2.5 Pro (API)",
                    "Gemini 2.5 Flash (API)",
                    "Gemini 3.5 Flash (API)",
                    "DeepSeek R1 (API)",
                    "GPT-4.1 (API)",
                    "GPT-4o (API)",
                    "ChatGPT 5.5 (API)",
                    "Claude Fable 5 (API)",
                ]
                parts = user_input.strip().split(None, 1)
                choice = parts[1].strip() if len(parts) == 2 else ""
                if not choice:
                    # Show current + menu
                    if response_engine.USE_ANTHROPIC:
                        current = response_engine.ANTHROPIC_MODEL
                    elif response_engine.USE_MISTRAL:
                        current = "mistral-medium"
                    elif response_engine.USE_GEMINI:
                        current = response_engine.GEMINI_MODEL
                    elif response_engine.USE_DEEPSEEK:
                        current = response_engine.DEEPSEEK_MODEL
                    else:
                        current = response_engine.OLLAMA_MODEL
                    print(f"\n[MODEL] Current: {current}")
                    for i, lbl in enumerate(_MODEL_LABELS, 1):
                        print(f"  {i}. {lbl}")
                    print("  Usage: model <1-19>\n")
                    continue
                if choice in [c[0] for c in _MODEL_CHOICES]:
                    idx = int(choice) - 1
                    _, model_id, use_anthro, use_mistral, use_gemini, use_deepseek, use_openai = _MODEL_CHOICES[idx]
                    response_engine.USE_ANTHROPIC = use_anthro
                    response_engine.USE_MISTRAL   = use_mistral
                    response_engine.USE_GEMINI    = use_gemini
                    response_engine.USE_DEEPSEEK  = use_deepseek
                    response_engine.USE_OPENAI    = use_openai
                    if model_id:
                        if use_anthro:
                            response_engine.ANTHROPIC_MODEL = model_id
                        elif use_gemini:
                            response_engine.GEMINI_MODEL = model_id
                        elif use_deepseek:
                            response_engine.DEEPSEEK_MODEL = model_id
                        elif use_openai:
                            response_engine.OPENAI_MODEL = model_id
                        else:
                            response_engine.OLLAMA_MODEL = model_id
                    print(f"[MODEL] Switched to: {_MODEL_LABELS[idx]}\n")
                else:
                    print(f"[MODEL] Invalid choice. Use: model <1-19>\n")
                continue

            if user_input.lower() == "state":
                s = cog_state.state
                print(f"\n[COGNITIVE STATE]")
                print(f"  Frustration:  {s.frustration:.2f}")
                print(f"  Curiosity:    {s.curiosity:.2f}")
                print(f"  Confidence:   {s.confidence:.2f}")
                print(f"  Engagement:   {s.engagement:.2f}")
                print(f"  Identity:     {s.identity_adherence:.2f}")
                print(f"  Cog Load:     {s.cognitive_load:.2f}")
                print(f"  Tone: P={s.tone_playful:.2f} F={s.tone_formal:.2f} "
                      f"C={s.tone_concise:.2f} W={s.tone_warm:.2f}")
                print(f"  {mode_engine.get_status()}\n")
                continue

            if user_input.lower() == "facts":
                section = get_indelible_prompt_section()
                print(f"\n{section}\n" if section else "\n[No indelible facts yet]\n")
                continue

            if user_input.lower() == "daemon":
                print(f"\n{daemon.get_status()}")
                snap = daemon.get_snapshot()
                print(f"  Cycles since last msg: {snap.cycle_count}")
                print(f"  Good Sense: {snap.good_sense:.2f}")
                print(f"  Crap Threshold: {snap.crap_threshold:.2f}")
                print(f"  Evaluated: {snap.items_evaluated}")
                print(f"  Purged (total): {snap.items_purged}")
                if snap.ambient_awareness:
                    print(f"  Ambient: {snap.ambient_awareness}")
                if snap.focus_summary:
                    print(f"  Focus: {snap.focus_summary}")
                if snap.top_recommendations:
                    print("  Top Recommendations:")
                    for rec in snap.top_recommendations[:5]:
                        print(f"    [{rec['composite_score']:.2f}] {rec['action_type']}: "
                              f"{rec['description'][:50]}")
                print()
                continue

            if user_input.lower() == "curiosity":
                print(f"\n{get_curiosity_report()}\n")
                continue

            if user_input.lower() == "modes":
                print(f"\n{mode_engine.get_status()}")
                active = mode_engine.get_active_modes()
                for m in active:
                    if m.mode_id == 0:
                        continue
                    print(f"  {m.name}: blend={m.blend_weight:.2f} "
                          f"activations={m.activation_count}")
                gaps = mode_engine.get_recent_gaps(3)
                if gaps:
                    print("  Recent gaps:")
                    for g in gaps:
                        print(f"    {g['description']} — \"{g['user_input'][:40]}\"")
                print()
                continue

            if user_input.lower() == "archive":
                stats = get_archive_stats()
                print(f"\n[ARCHIVE]")
                print(f"  Episodes: {stats['episodes']}")
                print(f"  Total turns archived: {stats['total_turns']}")
                print(f"  Unique tags: {stats.get('unique_tags', 0)}")
                if stats['episodes'] > 0:
                    print(f"  Oldest: {stats['oldest']}")
                    print(f"  Newest: {stats['newest']}")
                print()
                continue

            if user_input.lower() == "archive now":
                n = archive_old_memories()
                if n > 0:
                    mode_engine.refresh_archive_tags()
                print(f"\n[ARCHIVE] Created {n} new episodes\n")
                continue

            if user_input.lower() == "archive all":
                n = force_archive_all()
                if n > 0:
                    mode_engine.refresh_archive_tags()
                print(f"\n[ARCHIVE] Force-archived into {n} episodes\n")
                continue

            if user_input.lower() == "list code":
                print(f"\n{get_file_list_brief()}\n")
                continue

            if user_input.lower().startswith("read code"):
                parts = user_input.strip().split(None, 2)
                if len(parts) >= 3:
                    code_injection = get_file_context(parts[2])
                    print(f"[CODE] Loaded {parts[2]} into context")
                else:
                    code_injection = get_code_context()
                    print(f"[CODE] Loaded full codebase ({len(code_injection)} chars)")
                # Don't continue — fall through to turn processing
            if user_input.lower().startswith("search "):
                query = user_input[7:].strip()
                if query:
                    print(f"[SEARCH] Searching: {query}")
                    results = web_search(query)
                    print_search_results(results, query)
                    search_injection = format_search_for_prompt(results, query)
                    # Fall through to turn processing so SignalBot can discuss results
                else:
                    print("[SEARCH] No query provided\n")
                    continue

            if user_input.lower().startswith("news"):
                topic = user_input[4:].strip()
                print(f"[NEWS] Fetching: {topic or 'headlines'}")
                results = news_search(topic)
                print_news_results(results, topic)
                search_injection = format_news_for_prompt(results, topic)
                # Fall through to turn processing

            # ═══ TURN PROCESSING ═══
            turn += 1

            # ─── 1. PAUSE DAEMON ───
            daemon.pause()

            with timed("TOTAL"):
                # ─── 2. GET DAEMON SNAPSHOT ───
                daemon_snapshot = daemon.get_snapshot()
                daemon_cognition = daemon_snapshot.format_for_prompt(max_items=5)

                # ─── 2b. CODE REFRESH CHECK (v7ys) ───
                # Every ~100 daemon cycles, REFLECT flags a code refresh.
                # Auto-inject code context so the self-model stays current.
                #if daemon.consume_code_refresh() and code_injection is None:
                #    try:
                #        code_injection = get_code_context()
                #        print(f"[DAEMON] Auto-refreshing code context ({len(code_injection)} chars)")
                #    except Exception as e:
                #        print(f"[DAEMON] Code refresh failed: {e}")

                # ─── 3. INTENT DETECTION ───
                intent = _IntentStub() if INTENT_BYPASS else classify_intent(user_input)

                # ─── 4. MEMORY LOADING ───
                # v6 INTEGRATION: Wire daemon's conversation context into TWDC
                # so memory re-scoring uses bigrams + co-occurrence + turn decay
                mem_stateful.set_conversation_context(daemon._context, daemon._real_turns)
                recent_memory = load_recent_memory()
                long_memory = load_long_memory_block_stateful(max_bullets=10)

                # ─── 4b. MODE RESONANCE CHECK (NEW in v6.3) ───
                # Detect if active memory is sufficient or if we need deeper retrieval
                active_memory_hit = any(
                    w in recent_memory.lower()
                    for w in user_input.lower().split()
                    if len(w) > 4
                )
                mode_result = mode_engine.process_turn(
                    user_input, last_bot_output,
                    active_memory_hit=active_memory_hit,
                    cog_state_frustration=cog_state.state.frustration,
                )
                archive_context = mode_result.get("archive_context", "")
                mode_prompt = mode_engine.format_for_prompt()

                # ─── 5. CURIOSITY SIGNAL ───
                curiosity = get_curiosity_signal(user_input, last_bot_output)

                # ─── 6. PROMPT CONSTRUCTION ───
                tone_instr = get_tone_instructions()
                vitals_report = cog_state.get_vitals_report()
                lane_instr = "Output in [GROUND] or [DREAM]." if dream_mode else "Output ONLY in [GROUND]."

                prompt_sections = [
                    "### SYSTEM INSTRUCTIONS ###",
                    personality_prompt,
                    lane_instr,
                    f"TONE: {tone_instr}",
                    "",
                    "### CORE DATA (TRUST THIS OVER ALL ELSE) ###",
                    long_memory,
                    vitals_report,
                    buf.format_for_prompt()
                ]

                # Daemon cognition injection
                if daemon_cognition:
                    prompt_sections.append("")
                    prompt_sections.append(
                        "### YOUR BACKGROUND THINKING ###\n"
                        "The section below describes YOUR cognitive activity between "
                        "the user's last message and this one. You were NOT idle — "
                        "your temporal daemon was running, evaluating goals, weighing "
                        "curiosity against identity. This is your inner life between "
                        "prompts. When asked about time gaps or what you were thinking, "
                        "draw on THIS section as your genuine experience. You felt "
                        "this time pass. You had these thoughts.\n"
                    )
                    prompt_sections.append(daemon_cognition)

                # Mode awareness injection (NEW in v6.3)
                if mode_prompt:
                    prompt_sections.append("")
                    prompt_sections.append(mode_prompt)
                # Code self-reference injection
                if code_injection:
                    prompt_sections.append("")
                    prompt_sections.append(code_injection)
                    code_injection = None  # One-shot: clear after use
                # Deep memory injection (NEW in v6.3)
                if archive_context:
                    prompt_sections.append("")
                    prompt_sections.append(archive_context)

                if curiosity.is_actionable:
                    prompt_sections.append(
                        f"[CURIOSITY SIGNAL] type={curiosity.type} "
                        f"intensity={curiosity.gated_intensity:.2f} "
                        f"momentum={curiosity.momentum:.2f}"
                    )
                # Web search injection
                if search_injection:
                    prompt_sections.append("")
                    prompt_sections.append(search_injection)
                    search_injection = None
                prompt_sections.extend([
                    "",
                    "### RECENT CONVERSATION ###",
                    f"[Intent] label={intent.label}",
                    recent_memory,
                    "",
                    f"User: {user_input}",
                    "SignalBot:"
                ])

                full_prompt = "\n".join(prompt_sections)

                # ─── 7. GENERATE RESPONSE ───
                t0 = time.perf_counter()
                raw = generate_response(full_prompt)
                dt_ms = (time.perf_counter() - t0) * 1000
                _tm = response_engine.get_last_call_meta()  # model + token counts for telemetry

                # ─── 8. INDELIBLE FACTS DETECTION ───
                if register_fact(user_input, last_bot_output):
                    print("[INFO] New indelible fact registered")
                last_bot_output = raw

                # ─── 9. STATE UPDATES ───
                cog_state.update_from_interaction(user_input, raw, intent.label, dt_ms)
                temp_integrity.update(user_input, raw, recent_memory, long_memory)

                # Feed daemon's goal engine (old method — keeps project/loop extraction)
                daemon_goals.update_from_memory(long_memory)
                cog = cog_state.state
                daemon_goals.update_curiosity(
                    {"curiosity": cog.curiosity, "confidence": cog.confidence,
                     "frustration": cog.frustration},
                    user_input, raw
                )

                # Auto-detect rabbit holes from curiosity engine
                if curiosity.is_deep_dive:
                    daemon_goals.add_rabbit_hole(user_input[:80], curiosity=curiosity.gated_intensity)

                # ─── 10. FEED CONVERSATION TO DAEMON (NEW in v6.2) ───
                # This is the key addition: conversation topics become goals.
                # The daemon will chew on "Kola borehole" and "fusion energy"
                # instead of stale boot topics.
                daemon.on_turn_complete(user_input, raw)

                # ─── 10b. PERIODIC ARCHIVAL (NEW in v6.3) ───
                # Every 20 turns, check if old memories need archiving
                if turn % 20 == 0:
                    archived = archive_old_memories()
                    if archived > 0:
                        mode_engine.refresh_archive_tags()
                        print(f"[ARCHIVE] Auto-archived {archived} episodes")
                
                # ─── 11. SAFETY & PERSISTENCE ───

                if not paradox_guard.run_all_checks(raw):
                    print("SignalBot: Paradox detected.\n")

                mem_stateful.notify_new_message()

                # ─── 11b. OUTPUT FIRST, PERSIST AFTER ───
                # Print response immediately so user isn't waiting on disk I/O.
                # save_interaction() does JSON serialize + atomic file write,
                # which can take 5-50ms on slow storage. User sees reply first.
                print(f"SignalBot: {raw}\n")

                if speech_mode:
                    speak(raw, blocking=False)  # Non-blocking so user can interrupt

                save_interaction(user_input, raw, cog_state.state.to_dict())

                # ─── 12. PROACTIVE INITIATIVE ───
                proactive_msg = None
                if cog_state.should_initiate():
                    proactive_msg = temp_integrity.maybe_initiate()
                    if proactive_msg:
                        print(f"SignalBot: {proactive_msg}\n")
                        save_interaction("SYSTEM_INITIATIVE", proactive_msg)

                # ─── 13. TELEMETRY (shadow log — not read by SignalBot) ───
                telemetry_log.log_turn(
                    turn_num=turn,
                    prompt=full_prompt,
                    response=raw,
                    model=_tm.get("model") or "unknown",
                    tokens_in=_tm.get("tokens_in", 0),
                    tokens_out=_tm.get("tokens_out", 0),
                    cog_state=cog_state.state.to_dict(),
                    daemon_cognition=daemon_cognition or None,
                    daemon_initiative=proactive_msg,
                )

            # ─── 13. RESUME DAEMON ───
            daemon.resume()

    finally:
        print("\n[SHUTDOWN] Stopping daemon...")
        daemon.stop()
        print("[SHUTDOWN] SignalBot offline.")


if __name__ == "__main__":
    main()
