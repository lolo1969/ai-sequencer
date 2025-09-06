#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random, argparse, re, threading, queue, math, time
import mido
from openai import OpenAI

client = OpenAI()

# ================================
# Default configuration (CLI can override)
# ================================
DEFAULT_MIDI_OUT_PORT = "IAC Driver Bus 1"
BPM            = 118.0
LOOP_BARS      = 2
TIME_SIG       = (4, 4)
VOICES         = 4
USE_CHANNELS   = [1, 2, 3, 4][:VOICES]
VELOCITY_RANGE = (72, 108)
GLOBAL_KEY     = "C"
GLOBAL_MODE    = "Dorian"
PITCH_RANGE    = (48, 84)              # general fallback range
VOICE_RANGES   = {                     # optional per-channel ranges (keeps roles musical)
    1: (36, 55),  # bass
    2: (48, 72),  # mid
    3: (55, 84),  # lead
    4: (60, 96)   # extra
}
HUMANIZE_T     = 0.006
MODEL_NAME     = "gpt-4o-mini"
TEMPERATURE    = 0.6
MUTATION_AMOUNT = 0.35
SCENE_HOLD_LOOPS = 4         # vorher: 12
MICRO_CHANGE_FRACTION = 0.3  # vorher: 0.05
PITCH_STEP_LIMIT = 1
LOCK_STARTS = True

STYLE_PROMPT = (
    "Erzeuge hypnotische, minimalistische Patterns im Stil von Caterina Barbieri: "
    "Repetition mit subtilen Variationen und polyrhythmischen Überlagerungen. "
    "Keine Drums, nur Noten (Pitch/Gate)."
)

# ================================
# Scales & utilities
# ================================
SCALE_STEPS = {
    "ionian":      [0,2,4,5,7,9,11],
    "dorian":      [0,2,3,5,7,9,10],
    "phrygian":    [0,1,3,5,7,8,10],
    "lydian":      [0,2,4,6,7,9,11],
    "mixolydian":  [0,2,4,5,7,9,10],
    "aeolian":     [0,2,3,5,7,8,10],
    "locrian":     [0,1,3,5,6,8,10],
}
SCALE_ALIASES = {"major":"ionian","minor":"aeolian"}

NOTE_TO_SEMITONE = {
    "c":0,"c#":1,"db":1,"d":2,"d#":3,"eb":3,"e":4,"fb":4,"e#":5,"f":5,
    "f#":6,"gb":6,"g":7,"g#":8,"ab":8,"a":9,"a#":10,"bb":10,"b":11,"cb":11,"b#":0
}

def note_to_midi(s):
    """ 'C4' -> 60, 'D#3' -> 51, or '60' -> 60 """
    s = str(s).strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    m = re.fullmatch(r"([A-Ga-g][#b]?)(-?\d+)", s)
    if not m:
        raise ValueError(f"Ungültiger Root: {s}")
    name = m.group(1).lower()
    octv = int(m.group(2))
    sem  = NOTE_TO_SEMITONE[name]
    return 12*(octv+1) + sem  # MIDI C4=60

def scale_degree_to_semitones(deg, steps):
    n = len(steps)
    octave = deg // n
    idx    = deg % n
    return 12*octave + steps[idx]

def parse_seed_list(txt):
    parts = re.split(r"[,\s]+", txt.strip())
    return [int(p) for p in parts if p != ""]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def beats_per_bar(ts):
    num, den = ts
    return num * (4.0/den)

def sec_per_beat(bpm):
    return 60.0 / bpm

def quantize(val_beats, grid):
    return round(val_beats / grid) * grid

# ================================
# Seed / Evolution
# ================================
def mutate_seed(seed_notes, mut, fraction=MICRO_CHANGE_FRACTION):
    """Sparse, gentle variation on an existing note list."""
    if not seed_notes:
        return []
    out = []
    k = max(1, int(len(seed_notes) * fraction))
    idx_to_edit = set(random.sample(range(len(seed_notes)), k))
    # Debug-Ausgabe: Zeige wie viele Noten mutiert werden
    print(f"[DEBUG] Mutiere {len(idx_to_edit)} von {len(seed_notes)} Noten (Mutation={mut}, Fraction={fraction})")
    for i, n in enumerate(seed_notes):
        note = dict(n)
        if i in idx_to_edit:
            if random.random() < mut*0.9:
                note["midi"] = clamp(
                    note["midi"] + random.choice([-PITCH_STEP_LIMIT, 0, PITCH_STEP_LIMIT]),
                    PITCH_RANGE[0], PITCH_RANGE[1]
                )
            if (not LOCK_STARTS) and random.random() < mut*0.4:
                note["start_beats"] = max(0.0, note["start_beats"] + random.choice([0.0, 0.25, -0.25]))
            if random.random() < mut*0.3:
                note["dur_beats"] = max(0.125, note["dur_beats"] + random.choice([0.0, 0.25, -0.25]))
            if random.random() < mut*0.2:
                note["velocity"] = clamp(note["velocity"] + random.choice([-4, 0, 4]),
                                         VELOCITY_RANGE[0], VELOCITY_RANGE[1])
        out.append(note)
    return out

def build_seed_from_sequence(seq, root_midi, mode_name, seed_mode="degree",
                             step_dur=0.5, note_dur=0.75, rest_amount=0.15,
                             channel_step_durs=None, loop_beats=None):
    """Builds initial pattern; channel-specific step durations supported."""
    steps = SCALE_STEPS[mode_name.lower()]
    notes = []
    num_channels = len(channel_step_durs) if channel_step_durs else len(USE_CHANNELS)
    for ch_i in range(num_channels):
        # Fix: always use a copy of the sequence for each channel, and do not shuffle in-place
        # Also: ensure seq_ch is not empty and contains more than one unique value
        if isinstance(seq[0], list) and ch_i < len(seq):
            seq_ch = list(seq[ch_i])
        elif not isinstance(seq, list) or not isinstance(seq[0], list):
            seq_ch = list(seq)
        else:
            seq_ch = []
        # --- Fix: always use the correct root for channel 1 (bass) ---
        # If channel_roots is not set, set root_for_channel for channel 1 to a typical bass root (e.g. G2 = 43)
        if 'channel_roots' in globals() and channel_roots and ch_i < len(channel_roots):
            root_for_channel = channel_roots[ch_i]
        elif USE_CHANNELS[ch_i] == 1 and root_midi > 55:
            # If root is high (e.g. C4), transpose down to bass octave for channel 1
            root_for_channel = root_midi - 24  # two octaves down
        else:
            root_for_channel = root_midi
        # ...existing code...
        # (rest unchanged, just use root_for_channel below)
        step_dur_ch = channel_step_durs[ch_i] if channel_step_durs else step_dur
        max_beats = loop_beats if loop_beats else 8.0
        seq_len = len(seq_ch) if isinstance(seq_ch, list) else 0
        step_count = 0
        steps_list = []
        t = 0.0
        while t < max_beats:
            val_idx = step_count % seq_len if seq_len > 0 else None
            val = seq_ch[val_idx] if val_idx is not None else None
            is_note = (val is not None) and (random.random() >= rest_amount)
            steps_list.append((t, val if is_note else None))
            t += step_dur_ch
            step_count += 1
        note_times = [tt for tt, v in steps_list if v is not None]
        note_vals  = [v for tt, v in steps_list if v is not None]
        for i, (tt, val) in enumerate(zip(note_times, note_vals)):
            if val is None:
                continue
            try:
                val_int = int(val)
            except Exception:
                continue
            off = scale_degree_to_semitones(val_int, steps) if seed_mode == "degree" else val_int
            rng = VOICE_RANGES.get(USE_CHANNELS[ch_i], PITCH_RANGE)
            midi = clamp(root_for_channel + off, *rng)
            next_t = note_times[i + 1] if i + 1 < len(note_times) else max_beats
            gate_dur = max(0.05, min(note_dur, step_dur_ch * 0.8, next_t - tt - 0.01))
            notes.append({
                "start_beats": tt,
                "dur_beats":   gate_dur,
                "midi":        midi,
                "velocity":    random.randint(*VELOCITY_RANGE),
                "channel":     USE_CHANNELS[ch_i % len(USE_CHANNELS)]
            })
    return notes

# ================================
# OpenAI — prompts & calls
# ================================
SYSTEM_MSG = (
    "You are a strict MIDI pattern generator and you ONLY respond with JSON.\n"
    "Format:\n"
    "{ \"notes\": [ {\"start_beats\": float>=0, \"dur_beats\": float>0, "
    "\"midi\": int 0..127, \"velocity\": int 1..127, \"channel\": int 1..16} ] }\n"
    "No explanations, no comments — only JSON."
)

def read_extra_prompt(filepath="prompt.txt"):
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def build_user_prompt(prev_notes, loop_beats):
    extra_prompt = read_extra_prompt()
    style = STYLE_PROMPT if not extra_prompt else STYLE_PROMPT + "\n" + extra_prompt
    context = {
        "previous_loop": {"loop_beats": loop_beats, "notes": prev_notes},
        "constraints": {
            "voices": VOICES, "allow_channels": USE_CHANNELS, "bpm": BPM,
            "key": GLOBAL_KEY, "mode": GLOBAL_MODE,
            "pitch_min": PITCH_RANGE[0], "pitch_max": PITCH_RANGE[1],
            "velocity_min": VELOCITY_RANGE[0], "velocity_max": VELOCITY_RANGE[1]
        },
        "style": style,
        "variation_policy": {
            "keep_ratio": 0.95,
            "max_changes": 0.05,
            "max_pitch_step": PITCH_STEP_LIMIT,
            "lock_starts": LOCK_STARTS,
            "lock_durations": True
        },
        "mutation": f"Please evolve the pattern very subtly (mutation={MUTATION_AMOUNT}). "
                    "No sudden jumps, only minimal deviations over many loops."
    }
    return json.dumps(context)

def _parse_json_strict(txt, prev_notes):
    try:
        data = json.loads(txt)
        notes = data.get("notes", [])
        cleaned = []
        for n in notes:
            start = max(0.0, float(n["start_beats"]))
            dur   = float(n["dur_beats"])
            if dur <= 0:
                continue
            midi  = int(n["midi"])
            vel   = int(n["velocity"])
            ch    = int(n["channel"])
            # light quantization keeps the groove tight
            start = quantize(start, 0.25)
            dur   = max(0.05, quantize(dur, 0.125))
            rng   = VOICE_RANGES.get(ch, PITCH_RANGE)
            midi  = clamp(midi, *rng)
            vel   = clamp(vel, VELOCITY_RANGE[0], VELOCITY_RANGE[1])
            if ch not in USE_CHANNELS:
                ch = random.choice(USE_CHANNELS)
            cleaned.append({"start_beats": start,"dur_beats": dur,"midi": midi,"velocity": vel,"channel": ch})
        if cleaned:
            return cleaned
    except Exception:
        pass
    # fallback: mutate so the jam never stops
    return mutate_seed(prev_notes, max(0.2, MUTATION_AMOUNT))

def _json_block_from_text(txt):
    # find first { and last } — safer than greedy regex
    i = txt.find("{")
    j = txt.rfind("}")
    if i != -1 and j != -1 and j > i:
        return txt[i:j+1]
    return None

def chat_completion_json(messages, model=MODEL_NAME, temperature=TEMPERATURE, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=messages
            )
        except Exception as e:
            text = str(e).lower()
            if attempt < max_retries - 1 and any(t in text for t in ("429","rate","temporar","timeout","unavailable")):
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise

def request_next_loop(prev_notes, loop_beats):
    messages = [
        {"role":"system", "content": SYSTEM_MSG},
        {"role":"user",   "content": build_user_prompt(prev_notes, loop_beats)}
    ]
    try:
        resp = chat_completion_json(messages)
        txt = resp.choices[0].message.content
        return _parse_json_strict(txt, prev_notes)
    except Exception:
        # fallback: no response_format path
        resp = client.chat.completions.create(
            model=MODEL_NAME, temperature=TEMPERATURE, messages=messages
        )
        raw = resp.choices[0].message.content
        block = _json_block_from_text(raw)
        if block:
            return _parse_json_strict(block, prev_notes)
        return mutate_seed(prev_notes, max(0.3, MUTATION_AMOUNT))

def request_initial_params():
    prompt = (
        "You are a creative music assistant. "
        "Determine sensible initial parameters for a MIDI sequencer suitable for the following prompt. "
        "Respond ONLY with JSON, e.g.:\n"
        "{ \"voices\": 4, \"bpm\": 120, \"bars\": 2, \"key\": \"C\", \"mode\": \"Dorian\", \"seed\": \"0,2,4,1,6,2,4,2\" }\n"
        f"Prompt:\n{STYLE_PROMPT}\n"
    )
    try:
        resp = chat_completion_json([{"role":"user","content":prompt}])
        txt = resp.choices[0].message.content
        return json.loads(txt)
    except Exception as e:
        print("[WARN] AI parameter selection failed, using defaults.", e)
        return {}

# ================================
# MIDI
# ================================
def open_midi_out(preferred=None):
    names = mido.get_output_names()
    if names:
        print("Gefundene MIDI-Out-Ports:\n - " + "\n - ".join(names))
    else:
        print("Keine MIDI-Out-Ports gefunden.")
    if preferred and preferred in names:
        print(f"Öffne: {preferred}")
        return mido.open_output(preferred)
    for key in ("IAC", "IAC Driver", "IAC-Treiber", "Loop", "Virtual"):
        matches = [n for n in names if key.lower() in n.lower()]
        if matches:
            print(f"Öffne (Match '{key}'): {matches[0]}")
            return mido.open_output(matches[0])
    fallback_name = "AI Sequencer Out"
    print(f"Kein passender Port → erstelle virtuellen Port: '{fallback_name}'")
    print("Hinweis: In Ableton diesen Port als MIDI-Eingang aktivieren (Preferences → Link/MIDI).")
    return mido.open_output(fallback_name, virtual=True)

def reset_all_notes(midi_out):
    """All-notes-off (and pitchbend reset) on all channels."""
    for ch in range(16):
        for note in range(PITCH_RANGE[0], PITCH_RANGE[1]+1):
            midi_out.send(mido.Message('note_off', note=note, velocity=0, channel=ch))
        midi_out.send(mido.Message('pitchwheel', pitch=0, channel=ch))
    try:
        midi_out.send(mido.Message('reset'))
    except Exception:
        pass

# ================================
# Playback
# ================================
def schedule_events(notes, bpm):
    spb = sec_per_beat(bpm)
    events = []
    for n in notes:
        if n["dur_beats"] <= 0:
            continue
        on_t  = n["start_beats"] * spb
        off_t = (n["start_beats"] + n["dur_beats"]) * spb
        ch    = n["channel"] - 1
        # subtle per-note bend option on channel 2
        if (n["channel"] == 2) and (random.random() < 0.25):
            bend_val = random.randint(-200, 200)  # small ± range
            events.append(("pitchbend", on_t - 0.03, bend_val, ch))
            events.append(("pitchbend", off_t + 0.01, 0, ch))
        events.append(("on",  on_t,  n["midi"], ch, n["velocity"]))
        events.append(("off", off_t, n["midi"], ch, 0))
    # stable sort: by time, then pitchbend -> off -> on
    order = {"pitchbend":0, "off":1, "on":2}
    events.sort(key=lambda e: (e[1], order.get(e[0], 3)))
    return events

def play_loop(midi_out, events, t0, send_clock=False, bpm=120, loop_beats=8):
    # optional clock
    clock_thread = None
    clock_running = False
    if send_clock:
        midi_out.send(mido.Message('start'))
        clock_running = True
        def clock_worker():
            clock_interval = sec_per_beat(bpm) / 24.0
            loop_duration  = sec_per_beat(bpm) * loop_beats
            start_time = t0
            next_tick  = start_time
            while (time.monotonic() - start_time < loop_duration) and clock_running:
                now = time.monotonic()
                if now >= next_tick:
                    midi_out.send(mido.Message('clock'))
                    next_tick += clock_interval
                else:
                    time.sleep(min(0.001, next_tick - now))
        clock_thread = threading.Thread(target=clock_worker, daemon=True)
        clock_thread.start()

    i = 0
    while i < len(events):
        kind, t_abs, *params = events[i]
        now = time.monotonic() - t0
        sleep = t_abs - now
        if sleep > 0:
            time.sleep(sleep)
        jitter = random.uniform(-HUMANIZE_T, HUMANIZE_T)
        if jitter > 0:
            time.sleep(jitter)

        if kind == "pitchbend":
            bend_val, ch = params
            msg = mido.Message('pitchwheel', pitch=bend_val, channel=ch)
        else:
            note, ch, vel = params
            if kind == 'on':
                # safety: kill duplicates before re-on
                midi_out.send(mido.Message('note_off', note=note, velocity=0, channel=ch))
            msg = mido.Message('note_on' if kind=='on' else 'note_off',
                               note=note, velocity=vel, channel=ch)
        midi_out.send(msg)
        i += 1

    if send_clock and clock_thread:
        clock_running = False
        clock_thread.join()

# ================================
# Prefetch worker (background)
# ================================
def prefetch_worker(q, prev_ref, loop_beats, stop_ev):
    while not stop_ev.is_set():
        try:
            nxt = request_next_loop(prev_ref["notes"], loop_beats)
        except Exception as e:
            print("KI-Prefetch Fehler, nutze Mutation:", e)
            nxt = mutate_seed(prev_ref["notes"], max(0.3, MUTATION_AMOUNT))
        # keep only freshest without touching q.mutex
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(nxt)
        except queue.Full:
            pass
        stop_ev.wait(0.01)

# ================================
# CLI / main
# ================================
def main():
    global GLOBAL_MODE, VOICES, USE_CHANNELS, BPM, LOOP_BARS, MUTATION_AMOUNT, STYLE_PROMPT, GLOBAL_KEY, TEMPERATURE

    file_prompt = read_extra_prompt()
    if file_prompt:
        print(f"Prompt aus prompt.txt:\n{file_prompt}\n")
        STYLE_PROMPT = file_prompt

    p = argparse.ArgumentParser(description="AI → MIDI Jammer")
    p.add_argument("--seed", type=str, default=None, help="Start-Seed: '0,2,4,1,6,2,4,2'")
    p.add_argument("--seed-mode", type=str, choices=["degree","semitone"], default="degree")
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--mode", type=str, default=None)
    p.add_argument("--voices", type=int, default=None)
    p.add_argument("--bpm", type=float, default=None)
    p.add_argument("--bars", type=int, default=None)
    p.add_argument("--port", type=str, default=DEFAULT_MIDI_OUT_PORT)
    p.add_argument("--prefetch", type=int, choices=[0,1], default=1)
    p.add_argument("--rest-amount", type=float, default=0.15)
    p.add_argument("--send-clock", type=int, choices=[0,1], default=1)
    p.add_argument("--channel-divs", type=str, default=None,
                   help="Kommagetrennt: z.B. '4,8,16,8' (Viertel/Achtel/Sechzehntel/Achtel)")
    p.add_argument("--mutation", type=float, default=MUTATION_AMOUNT)
    p.add_argument("--channel-roots", type=str, default=None,
                   help="Kommagetrennte Root-Noten pro Kanal, z.B. 'C2,G3,E4,C3'")
    p.add_argument("--seed-rng", type=int, default=None, help="Deterministische Random-Seed")
    p.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperatur für KI (0.0-2.0)")
    args = p.parse_args()

    # validate
    if args.bpm is not None and not (20 <= args.bpm <= 300):
        p.error("--bpm must be between 20 and 300")
    if args.bars is not None and args.bars <= 0:
        p.error("--bars must be >= 1")
    if args.rest_amount < 0 or args.rest_amount > 1:
        p.error("--rest-amount must be 0..1")
    if args.seed_rng is not None:
        random.seed(args.seed_rng)

    # ask KI for defaults if nothing provided
    if (args.seed is None and args.root is None and args.mode is None and
        args.voices is None and args.bpm is None and args.bars is None):
        print("[INFO] Keine Parameter übergeben – KI bestimmt sinnvolle Startparameter ...")
        ki_params = request_initial_params()
        BPM       = ki_params.get("bpm", BPM)
        LOOP_BARS = ki_params.get("bars", LOOP_BARS)
        VOICES    = ki_params.get("voices", VOICES)
        USE_CHANNELS[:] = [1,2,3,4][:VOICES]
        GLOBAL_KEY  = ki_params.get("key", GLOBAL_KEY)
        GLOBAL_MODE = ki_params.get("mode", GLOBAL_MODE)
        if "root" in ki_params and re.match(r"^[A-Ga-g][#b]?-?\d+$", str(ki_params["root"])):
            root_str = ki_params["root"]
        else:
            root_str = f"{GLOBAL_KEY}4"
        # NEU: Seed von KI holen, sonst Default
        seed_str = ki_params.get("seed", "0,2,4,1,6,2,4,2")
    else:
        BPM       = args.bpm if args.bpm is not None else BPM
        LOOP_BARS = args.bars if args.bars is not None else LOOP_BARS
        VOICES    = clamp(args.voices, 1, 4) if args.voices is not None else VOICES
        USE_CHANNELS[:] = [1,2,3,4][:VOICES]
        GLOBAL_MODE = args.mode if args.mode is not None else GLOBAL_MODE
        root_str    = args.root if args.root is not None else f"{GLOBAL_KEY}4"
        seed_str    = args.seed if args.seed is not None else "0,2,4,1,6,2,4,2"

    MUTATION_AMOUNT = clamp(args.mutation, 0.0, 1.0)
    TEMPERATURE = clamp(args.temperature, 0.0, 2.0)

    # resolve scale alias
    mode_lower = str(GLOBAL_MODE).lower()
    GLOBAL_MODE = SCALE_ALIASES.get(mode_lower, mode_lower)
    if GLOBAL_MODE not in SCALE_STEPS:
        raise ValueError(f"Unbekannte Skala: {GLOBAL_MODE}")

    try:
        root_midi = note_to_midi(root_str)
    except Exception as e:
        print(f"[WARN] Fehler beim Parsen von root '{root_str}', nutze 'C4': {e}")
        root_midi = note_to_midi("C4")

    loop_beats = beats_per_bar(TIME_SIG) * LOOP_BARS

    print(f"Starte Jam @ {BPM} BPM, {LOOP_BARS} Takte, {VOICES} Stimmen,"
          f" Root={root_str}, Mode={GLOBAL_MODE}, Prefetch={args.prefetch}")

    # MIDI out
    out = open_midi_out(preferred=args.port)
    reset_all_notes(out)

    # per-channel step durations
    channel_step_durs = None
    if args.channel_divs:
        divs = [max(1, int(x)) for x in args.channel_divs.split(",")]
        # KORREKTUR: Schrittweite = beats_per_bar / div
        beats_bar = beats_per_bar(TIME_SIG)
        channel_step_durs = [beats_bar / d for d in divs]
    # per-channel roots (optional)
    channel_roots = None
    if args.channel_roots:
        channel_roots = [note_to_midi(r.strip()) for r in args.channel_roots.split(",")]
    elif args.root and "," in args.root:
        channel_roots = [note_to_midi(r.strip()) for r in args.root.split(",")]
    if channel_roots:
        while len(channel_roots) < VOICES:
            channel_roots.append(channel_roots[0])
        channel_roots = channel_roots[:VOICES]

    # build seed
    if seed_str:
        seq  = parse_seed_list(seed_str)
        if channel_step_durs:
            seqs = [seq for _ in range(len(channel_step_durs))]
            seed = build_seed_from_sequence(
                seqs, root_midi, GLOBAL_MODE, seed_mode=args.seed_mode,
                rest_amount=args.rest_amount, channel_step_durs=channel_step_durs,
                loop_beats=loop_beats
            )
        else:
            seed = build_seed_from_sequence(
                seq, root_midi, GLOBAL_MODE, seed_mode=args.seed_mode,
                rest_amount=args.rest_amount, loop_beats=loop_beats
            )
        prev = seed[:]
    else:
        # quick chordy seed
        base = root_midi
        chord_int = [0,3,7,10][:VOICES]
        seed = []
        for ch_i in range(len(USE_CHANNELS)):
            step_dur_ch = channel_step_durs[ch_i] if channel_step_durs else 0.5
            max_beats = loop_beats
            offs = chord_int
            seq_len = len(offs)
            step_count = 0
            steps_list = []
            t = 0.0
            while t < max_beats:
                val_idx = step_count % seq_len if seq_len > 0 else None
                val = offs[val_idx] if val_idx is not None else None
                is_note = val is not None and (random.random() >= args.rest_amount)
                steps_list.append((t, val if is_note else None))
                t += step_dur_ch
                step_count += 1
            note_times = [tt for tt, v in steps_list if v is not None]
            note_vals  = [v  for tt, v in steps_list if v is not None]
            rng = VOICE_RANGES.get(USE_CHANNELS[ch_i], PITCH_RANGE)
            for i, (tt, val) in enumerate(zip(note_times, note_vals)):
                midi = clamp(base + val, *rng)
                next_t = note_times[i + 1] if i + 1 < len(note_times) else max_beats
                gate_dur = max(0.05, min(step_dur_ch * 0.8, next_t - tt - 0.01))
                seed.append({
                    "start_beats": tt, "dur_beats": gate_dur,
                    "midi": midi,
                    "velocity": random.randint(*VELOCITY_RANGE),
                    "channel": USE_CHANNELS[ch_i % len(USE_CHANNELS)]
                })
        prev = seed[:]

    scene_loop_count = 0

    # Prefetch
    if args.prefetch == 1:
        q = queue.Queue(maxsize=1)
        stop_ev = threading.Event()
        prev_ref = {"notes": prev[:]}
        th = threading.Thread(target=prefetch_worker, args=(q, prev_ref, loop_beats, stop_ev), daemon=True)
        th.start()

    # main jam loop
    try:
        while True:
            next_notes = prev[:]  # default repeat
            if args.prefetch == 1:
                try:
                    ready = q.get_nowait()
                    if ready:
                        next_notes = ready
                except queue.Empty:
                    pass
            else:
                if scene_loop_count == 0:
                    try:
                        cand = request_next_loop(prev, loop_beats)
                        if cand:
                            next_notes = cand
                    except Exception as e:
                        print("KI-Abruf fehlgeschlagen – spiele identisch weiter:", e)

            events = schedule_events(next_notes, BPM)
            t0 = time.monotonic()
            play_loop(out, events, t0, send_clock=bool(args.send_clock), bpm=BPM, loop_beats=loop_beats)

            prev = next_notes

            scene_loop_count += 1
            if scene_loop_count >= SCENE_HOLD_LOOPS:
                prev = mutate_seed(prev, MUTATION_AMOUNT, fraction=MICRO_CHANGE_FRACTION)
                scene_loop_count = 0

            if args.prefetch == 1:
                prev_ref["notes"] = prev
                # nudge worker to refresh
                stop_ev.set(); stop_ev.clear()

    except KeyboardInterrupt:
        print("\n[INFO] Abbruch – alle Gates schließen.")
        reset_all_notes(out)
        # out.close()  # optional


if __name__ == "__main__":
    # mido.set_backend('mido.backends.rtmidi')  # uncomment if needed
    main()
