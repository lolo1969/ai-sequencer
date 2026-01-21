#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Evolving Jam - Ein KI-gesteuerter MIDI-Sequencer.

Dieser Sequencer generiert und evolviert MIDI-Patterns in Echtzeit
mithilfe von OpenAI's GPT-Modellen.
"""

import os
import json
import random
import argparse
import re
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, Any

import mido
from openai import OpenAI

client = OpenAI()

# ================================
# Konstanten (Magic Numbers vermeiden)
# ================================
MIDI_NOTE_C4 = 60
BASS_OCTAVE_THRESHOLD = 55
OCTAVE_SEMITONES = 12
TWO_OCTAVES = 24
ALL_NOTES_OFF_CC = 123
ALL_SOUND_OFF_CC = 120
MIDI_CLOCK_PPQN = 24  # Pulses per quarter note

# ================================
# Default configuration (CLI can override)
# ================================
DEFAULT_MIDI_OUT_PORT = "IAC Driver Bus 1"
DEFAULT_STYLE_PROMPT = (
    "Generate hypnotic, minimalist patterns in the style of Caterina Barbieri: "
    "Repetition with subtle variations and polyrhythmic overlays. "
    "No drums, only notes (Pitch/Gate)."
)


@dataclass
class Config:
    """Zentrale Konfiguration für den AI-Sequencer."""
    
    bpm: float = 118.0
    loop_bars: int = 2
    time_sig: tuple[int, int] = (4, 4)
    voices: int = 4
    velocity_range: tuple[int, int] = (72, 108)
    global_key: str = "C"
    global_mode: str = "Dorian"
    pitch_range: tuple[int, int] = (48, 84)
    voice_ranges: dict[int, tuple[int, int]] = field(default_factory=lambda: {
        1: (48, 67),  # bass (C3-G4) - höher für V/OCT, Oszillator eine Oktave runter
        2: (48, 72),  # mid
        3: (55, 84),  # lead
        4: (60, 96)   # extra
    })
    humanize_t: float = 0.006
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.6
    mutation_amount: float = 0.35
    scene_hold_loops: int = 4
    micro_change_fraction: float = 0.3
    pitch_step_limit: int = 1
    lock_starts: bool = True
    gate_length: float = 0.9  # Gate-Länge als Ratio der Step-Duration
    style_prompt: str = DEFAULT_STYLE_PROMPT
    midi_out_port: str = DEFAULT_MIDI_OUT_PORT
    
    @property
    def use_channels(self) -> list[int]:
        """Gibt die aktiven MIDI-Kanäle basierend auf der Stimmenanzahl zurück."""
        return [1, 2, 3, 4][:self.voices]


# Globale Config-Instanz (wird in main() konfiguriert)
config = Config()

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
    "maqam hijaz": [0,1,4,5,7,8,11],
    "maqam hijaz kar": [0,1,4,5,7,8,10],  # Korrigiert: Hijaz Kar hat b7 statt Maj7
}
SCALE_ALIASES = {"major":"ionian","minor":"aeolian"}

NOTE_TO_SEMITONE = {
    "c":0,"c#":1,"db":1,"d":2,"d#":3,"eb":3,"e":4,"fb":4,"e#":5,"f":5,
    "f#":6,"gb":6,"g":7,"g#":8,"ab":8,"a":9,"a#":10,"bb":10,"b":11,"cb":11,"b#":0
}

def note_to_midi(s: str | int) -> int:
    """
    Konvertiert einen Notennamen oder MIDI-Nummer zu einer MIDI-Notennummer.
    
    Args:
        s: Notenname (z.B. 'C4', 'D#3') oder MIDI-Nummer (z.B. 60)
    
    Returns:
        MIDI-Notennummer (0-127)
    
    Raises:
        ValueError: Wenn das Format ungültig ist
    
    Examples:
        >>> note_to_midi('C4')
        60
        >>> note_to_midi('D#3')
        51
        >>> note_to_midi(60)
        60
    """
    s = str(s).strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    m = re.fullmatch(r"([A-Ga-g][#b]?)(-?\d+)", s)
    if not m:
        raise ValueError(f"Invalid root: {s}")
    name = m.group(1).lower()
    octv = int(m.group(2))
    sem  = NOTE_TO_SEMITONE[name]
    return OCTAVE_SEMITONES * (octv + 1) + sem  # MIDI C4=60

def scale_degree_to_semitones(deg: int, steps: list[int]) -> int:
    """Konvertiert einen Skalengrad zu Halbtonschritten.
    
    Verwendet 1-basierte Skalengrade wie in der Musiktheorie:
    1 = Wurzel (Root), 3 = Terz, 5 = Quinte, etc.
    """
    deg_0 = deg - 1  # Konvertiere zu 0-basiert für Berechnung
    n = len(steps)
    octave = deg_0 // n
    idx    = deg_0 % n
    return OCTAVE_SEMITONES * octave + steps[idx]


def parse_seed_list(txt: str) -> list[int]:
    """Parst eine komma- oder leerzeichengetrennte Liste von Zahlen."""
    parts = re.split(r"[,\s]+", txt.strip())
    return [int(p) for p in parts if p != ""]


def clamp(v: float | int, lo: float | int, hi: float | int) -> float | int:
    """Begrenzt einen Wert auf ein Intervall [lo, hi]."""
    return max(lo, min(hi, v))


# ================================
# Arrangement System
# ================================
def get_active_channels(
    loop_count: int,
    total_voices: int,
    arrangement_mode: str,
    loops_per_stage: int = 4
) -> list[int]:
    """
    Bestimmt welche Kanäle im aktuellen Loop aktiv sind.
    
    Args:
        loop_count: Aktueller Loop-Zähler
        total_voices: Gesamtzahl der Stimmen
        arrangement_mode: 'off', 'build', 'wave', 'random'
        loops_per_stage: Wie viele Loops pro Arrangement-Stufe
    
    Returns:
        Liste der aktiven Kanal-Nummern (1-based)
    """
    all_channels = list(range(1, total_voices + 1))
    
    if arrangement_mode == "off":
        return all_channels
    
    stage = loop_count // loops_per_stage
    
    if arrangement_mode == "build":
        # Graduelles Hinzufügen: 1 → 1,2 → 1,2,3 → alle → repeat
        num_active = min((stage % total_voices) + 1, total_voices)
        return all_channels[:num_active]
    
    elif arrangement_mode == "wave":
        # Wellenform: aufbauen → abbauen → aufbauen
        cycle_length = (total_voices - 1) * 2  # z.B. 1,2,3,4,3,2,1,2,3,4...
        if cycle_length == 0:
            return all_channels
        pos = stage % cycle_length
        if pos < total_voices:
            num_active = pos + 1
        else:
            num_active = cycle_length - pos + 1
        return all_channels[:max(1, num_active)]
    
    elif arrangement_mode == "random":
        # Zufällige Kombination, aber mindestens 1 Kanal
        random.seed(loop_count // loops_per_stage)  # Deterministisch pro Stage
        num_active = random.randint(1, total_voices)
        active = random.sample(all_channels, num_active)
        random.seed()  # Reset random state
        return sorted(active)
    
    elif arrangement_mode == "subtract":
        # Umgekehrtes Build: alle → 3 → 2 → 1 → repeat
        num_active = total_voices - (stage % total_voices)
        return all_channels[:num_active]
    
    return all_channels


def filter_notes_by_channels(notes: list[dict], active_channels: list[int]) -> list[dict]:
    """Filtert Noten auf die aktiven Kanäle."""
    return [n for n in notes if n.get("channel") in active_channels]


def midi_to_scale_degree(midi: int, root_midi: int, scale_steps: list[int]) -> tuple[int, int]:
    """
    Konvertiert eine MIDI-Note zu Skalengrad und Oktave relativ zur Root.
    
    Returns:
        (degree, octave_offset) - degree ist 1-basiert (1=Root, 3=Terz, 5=Quinte)
    """
    semitones_from_root = midi - root_midi
    octave_offset = semitones_from_root // 12
    semitone_in_octave = semitones_from_root % 12
    
    # Finde den nächsten Skalengrad
    best_degree = 1
    best_dist = 12
    for i, step in enumerate(scale_steps):
        dist = abs(step - semitone_in_octave)
        if dist < best_dist:
            best_dist = dist
            best_degree = i + 1  # 1-basiert
    
    return (best_degree, octave_offset)


def scale_degree_to_midi(degree: int, octave_offset: int, root_midi: int, scale_steps: list[int]) -> int:
    """
    Konvertiert Skalengrad und Oktave zurück zu MIDI.
    
    Args:
        degree: 1-basierter Skalengrad (1=Root, 3=Terz, 5=Quinte)
        octave_offset: Oktavverschiebung relativ zur Root
        root_midi: Root-MIDI-Note
        scale_steps: Liste der Halbtonschritte in der Skala
    """
    deg_0 = (degree - 1) % len(scale_steps)
    extra_octaves = (degree - 1) // len(scale_steps)
    semitones = scale_steps[deg_0] + (octave_offset + extra_octaves) * 12
    return root_midi + semitones


def beats_per_bar(ts: tuple[int, int]) -> float:
    """Berechnet die Anzahl Beats pro Takt für eine Taktart."""
    num, den = ts
    return num * (4.0 / den)


def sec_per_beat(bpm: float) -> float:
    """Berechnet die Dauer eines Beats in Sekunden."""
    return 60.0 / bpm


def quantize(val_beats: float, grid: float) -> float:
    """Quantisiert einen Beat-Wert auf ein Raster."""
    return round(val_beats / grid) * grid

# ================================
# Seed / Evolution
# ================================
def mutate_seed(
    seed_notes: list[dict],
    mut: float,
    fraction: Optional[float] = None,
    root_midi: Optional[int] = None,
    mode_name: Optional[str] = None
) -> list[dict]:
    """
    Führt harmonisch koordinierte Variationen an einer bestehenden Notenliste durch.
    
    WICHTIG: Noten die gleichzeitig spielen werden zusammen mutiert,
    um harmonische Beziehungen (Terzen, Quinten etc.) zu erhalten.
    
    Args:
        seed_notes: Liste von Noten-Dictionaries
        mut: Mutationsstärke (0.0 - 1.0)
        fraction: Anteil der Zeitpunkte zu ändern (default: config.micro_change_fraction)
        root_midi: Root-MIDI-Note für skalenbasierte Mutation
        mode_name: Name der Skala (z.B. 'aeolian')
    
    Returns:
        Mutierte Notenliste mit erhaltenen Harmonien
    """
    if fraction is None:
        fraction = config.micro_change_fraction
    
    if not seed_notes:
        return []
    
    # Hole Skalen-Infos aus config wenn nicht übergeben
    if root_midi is None:
        # Versuche aus den Noten abzuleiten (nimm den niedrigsten Pitch als Referenz)
        root_midi = min(n["midi"] for n in seed_notes)
    if mode_name is None:
        mode_name = config.global_mode
    
    scale_steps = SCALE_STEPS.get(mode_name.lower(), SCALE_STEPS["aeolian"])
    
    # Gruppiere Noten nach Start-Zeit (mit Toleranz von 0.1 Beats)
    time_groups: dict[float, list[tuple[int, dict]]] = {}
    for i, n in enumerate(seed_notes):
        t = round(n["start_beats"] / 0.25) * 0.25  # Quantisiere auf Viertel
        if t not in time_groups:
            time_groups[t] = []
        time_groups[t].append((i, n))
    
    # Wähle zufällige Zeitpunkte für Mutation
    time_keys = list(time_groups.keys())
    k = max(1, int(len(time_keys) * fraction))
    times_to_mutate = set(random.sample(time_keys, min(k, len(time_keys))))
    
    print(f"[DEBUG] Harmonic mutation: {len(times_to_mutate)} of {len(time_keys)} time slots "
          f"(Mutation={mut}, Fraction={fraction})")
    
    out: list[dict] = [dict(n) for n in seed_notes]
    
    for t in times_to_mutate:
        group = time_groups[t]
        
        if random.random() < mut * 0.9:
            # Harmonische Mutation: Alle Noten zur gleichen Zeit um dieselbe
            # Halbton-Distanz verschieben (erhält Intervalle EXAKT!)
            
            # Wähle einen Skalengrad-Shift und berechne die Halbtöne
            degree_shift = random.choice([-1, 0, 1, 2, -2])
            if degree_shift == 0:
                continue  # Keine Änderung
            
            # Berechne Halbtöne für diesen Skalengrad-Shift
            # Nimm die erste Note als Referenz
            ref_midi = out[group[0][0]]["midi"]
            ref_degree, ref_octave = midi_to_scale_degree(ref_midi, root_midi, scale_steps)
            new_ref_midi = scale_degree_to_midi(ref_degree + degree_shift, ref_octave, root_midi, scale_steps)
            semitone_shift = new_ref_midi - ref_midi  # Diese Halbtöne für ALLE Noten
            
            # Berechne zuerst alle neuen MIDI-Werte
            new_midis: list[tuple[int, int]] = []  # (idx, new_midi)
            for idx, orig_note in group:
                note = out[idx]
                new_midi = note["midi"] + semitone_shift  # Gleiche Halbtöne für alle!
                new_midis.append((idx, new_midi))
            
            # Prüfe ob alle neuen Werte im erlaubten Bereich sind
            all_valid = True
            for idx, new_midi in new_midis:
                ch = out[idx].get("channel", 1)
                rng = config.voice_ranges.get(ch, config.pitch_range)
                if new_midi < rng[0] or new_midi > rng[1]:
                    all_valid = False
                    break
            
            # Nur anwenden wenn ALLE Noten der Gruppe im Range bleiben
            # (sonst würden Intervalle durch Clamping zerstört)
            if all_valid:
                for idx, new_midi in new_midis:
                    out[idx]["midi"] = new_midi
            # Wenn nicht valid: Noten bleiben unverändert, Intervall bleibt erhalten
        
        # Timing-Mutation (optional, nur wenn nicht gelockt)
        if (not config.lock_starts) and random.random() < mut * 0.4:
            time_shift = random.choice([0.0, 0.25, -0.25])
            for idx, _ in group:
                out[idx]["start_beats"] = max(0.0, out[idx]["start_beats"] + time_shift)
        
        # Velocity-Mutation (gemeinsam für die Gruppe)
        if random.random() < mut * 0.2:
            vel_shift = random.choice([-4, 0, 4])
            for idx, _ in group:
                out[idx]["velocity"] = clamp(
                    out[idx]["velocity"] + vel_shift,
                    config.velocity_range[0], config.velocity_range[1]
                )
    
    return out

def build_seed_from_sequence(
    seq: list | list[list],
    root_midi: int,
    mode_name: str,
    seed_mode: str = "degree",
    step_dur: float = 0.5,
    note_dur: float = 0.75,
    rest_amount: float = 0.15,
    channel_step_durs: Optional[list[float]] = None,
    loop_beats: Optional[float] = None,
    channel_roots: Optional[list[int]] = None
) -> list[dict]:
    """
    Erstellt ein initiales Pattern aus einer Sequenz.
    
    Args:
        seq: Sequenz von Skalengraden oder Halbtonschritten
        root_midi: Basis-MIDI-Note
        mode_name: Name der Skala (z.B. 'dorian')
        seed_mode: 'degree' für Skalengrade, 'semitone' für Halbtonschritte
        step_dur: Standard-Schrittdauer in Beats
        note_dur: Standard-Notendauer in Beats
        rest_amount: Anteil der Pausen (0.0 - 1.0)
        channel_step_durs: Optionale Schrittdauern pro Kanal
        loop_beats: Länge des Loops in Beats
        channel_roots: Optionale Root-Noten pro Kanal
    
    Returns:
        Liste von Noten-Dictionaries
    """
    steps = SCALE_STEPS[mode_name.lower()]
    notes: list[dict] = []
    use_channels = config.use_channels
    num_channels = len(channel_step_durs) if channel_step_durs else len(use_channels)
    
    # Validierung: Leere Sequenz abfangen
    if not seq:
        return notes
    
    for ch_i in range(num_channels):
        # Sequenz für diesen Kanal bestimmen
        if isinstance(seq[0], list) and ch_i < len(seq):
            seq_ch = list(seq[ch_i])
        elif not isinstance(seq, list) or not isinstance(seq[0], list):
            seq_ch = list(seq)
        else:
            seq_ch = []
        
        # Root-Note für diesen Kanal bestimmen
        if channel_roots and ch_i < len(channel_roots):
            root_for_channel = channel_roots[ch_i]
        elif use_channels[ch_i] == 1:
            # Bass-Kanal: Eine Oktave nach unten transponieren
            root_for_channel = root_midi - 12
        else:
            root_for_channel = root_midi
        
        step_dur_ch = channel_step_durs[ch_i] if channel_step_durs else step_dur
        max_beats = loop_beats if loop_beats else 8.0
        seq_len = len(seq_ch) if isinstance(seq_ch, list) else 0
        step_count = 0
        steps_list: list[tuple[float, int | None]] = []
        t = 0.0
        
        while t < max_beats:
            val_idx = step_count % seq_len if seq_len > 0 else None
            val = seq_ch[val_idx] if val_idx is not None else None
            is_note = (val is not None) and (random.random() >= rest_amount)
            steps_list.append((t, val if is_note else None))
            t += step_dur_ch
            step_count += 1
        
        note_times = [tt for tt, v in steps_list if v is not None]
        note_vals = [v for tt, v in steps_list if v is not None]
        
        for i, (tt, val) in enumerate(zip(note_times, note_vals)):
            if val is None:
                continue
            try:
                val_int = int(val)
            except Exception:
                continue
            
            off = scale_degree_to_semitones(val_int, steps) if seed_mode == "degree" else val_int
            rng = config.voice_ranges.get(use_channels[ch_i], config.pitch_range)
            midi = clamp(root_for_channel + off, *rng)
            next_t = note_times[i + 1] if i + 1 < len(note_times) else max_beats
            
            # Gate-Länge: Nutze config.gate_length als Ratio
            gate_dur = max(0.05, min(step_dur_ch * config.gate_length, next_t - tt - 0.01))
            
            notes.append({
                "start_beats": tt,
                "dur_beats": gate_dur,
                "midi": midi,
                "velocity": random.randint(*config.velocity_range),
                "channel": use_channels[ch_i % len(use_channels)]
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

def read_extra_prompt(filepath: str = "prompt.txt") -> str:
    """Liest einen zusätzlichen Prompt aus einer Datei."""
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def build_user_prompt(prev_notes: list[dict], loop_beats: float) -> str:
    """
    Erstellt den User-Prompt für die OpenAI-API.
    
    Args:
        prev_notes: Noten des vorherigen Loops
        loop_beats: Länge des Loops in Beats
    
    Returns:
        JSON-formatierter Prompt-String
    """
    extra_prompt = read_extra_prompt()
    style = config.style_prompt if not extra_prompt else config.style_prompt + "\n" + extra_prompt
    context = {
        "previous_loop": {"loop_beats": loop_beats, "notes": prev_notes},
        "constraints": {
            "voices": config.voices,
            "allow_channels": config.use_channels,
            "bpm": config.bpm,
            "key": config.global_key,
            "mode": config.global_mode,
            "pitch_min": config.pitch_range[0],
            "pitch_max": config.pitch_range[1],
            "velocity_min": config.velocity_range[0],
            "velocity_max": config.velocity_range[1],
            "quantize_grid": 0.25,
            "sync_rule": "All channels MUST have notes starting at the SAME beat positions (0.0, 0.25, 0.5, etc). Keep channels synchronized!"
        },
        "style": style,
        "variation_policy": {
            "keep_ratio": 0.95,
            "max_changes": 0.05,
            "max_pitch_step": config.pitch_step_limit,
            "lock_starts": config.lock_starts,
            "lock_durations": True,
            "preserve_rhythm": True
        },
        "mutation": f"Please evolve the pattern very subtly (mutation={config.mutation_amount}). "
                    "No sudden jumps, only minimal deviations over many loops. "
                    "CRITICAL: Do NOT change start_beats values - only change pitch/velocity!"
    }
    return json.dumps(context)

def _parse_json_strict(txt: str, prev_notes: list[dict]) -> list[dict]:
    """
    Parst JSON-Antwort und validiert/bereinigt die Noten.
    
    Fällt bei Fehlern auf Mutation zurück.
    """
    try:
        data = json.loads(txt)
        notes = data.get("notes", [])
        cleaned: list[dict] = []
        
        for n in notes:
            start = max(0.0, float(n["start_beats"]))
            dur = float(n["dur_beats"])
            if dur <= 0:
                continue
            midi = int(n["midi"])
            vel = int(n["velocity"])
            ch = int(n["channel"])
            
            # Light quantization keeps the groove tight
            start = quantize(start, 0.25)
            dur = max(0.05, quantize(dur, 0.125))
            rng = config.voice_ranges.get(ch, config.pitch_range)
            midi = clamp(midi, *rng)
            vel = clamp(vel, config.velocity_range[0], config.velocity_range[1])
            
            if ch not in config.use_channels:
                ch = random.choice(config.use_channels)
            
            cleaned.append({
                "start_beats": start,
                "dur_beats": dur,
                "midi": midi,
                "velocity": vel,
                "channel": ch
            })
        
        if cleaned:
            # Sortiere nach Kanal und start_beats für konsistentes Playback
            cleaned.sort(key=lambda n: (n["channel"], n["start_beats"]))
            
            # Prüfe ob alle Kanäle vertreten sind - wenn nicht, ergänze aus prev_notes
            channels_in_cleaned = set(n["channel"] for n in cleaned)
            missing_channels = set(config.use_channels) - channels_in_cleaned
            
            if missing_channels and prev_notes:
                # Ergänze fehlende Kanäle aus dem vorherigen Pattern
                for ch in missing_channels:
                    ch_notes = [n for n in prev_notes if n.get("channel") == ch]
                    if ch_notes:
                        cleaned.extend(ch_notes)
                        print(f"[DEBUG] Kanal {ch} von AI vergessen - aus vorherigem Pattern ergänzt")
            
            return cleaned
    except Exception:
        pass
    
    # Fallback: mutate so the jam never stops
    return mutate_seed(prev_notes, max(0.2, config.mutation_amount))

def _json_block_from_text(txt: str) -> Optional[str]:
    """
    Extrahiert einen JSON-Block aus einem Text.
    
    Verwendet json.JSONDecoder für robusteres Parsing bei
    verschachtelten Strukturen.
    
    Args:
        txt: Text, der JSON enthalten könnte
    
    Returns:
        JSON-String oder None
    """
    # Versuche zuerst mit JSONDecoder (robuster)
    decoder = json.JSONDecoder()
    txt = txt.strip()
    
    # Suche nach dem ersten '{'
    for i, char in enumerate(txt):
        if char == '{':
            try:
                obj, end_idx = decoder.raw_decode(txt[i:])
                return json.dumps(obj)
            except json.JSONDecodeError:
                continue
    
    # Fallback: Einfache Suche nach { und }
    i = txt.find("{")
    j = txt.rfind("}")
    if i != -1 and j != -1 and j > i:
        return txt[i:j+1]
    
    return None

def chat_completion_json(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries: int = 3
) -> Any:
    """
    Führt einen Chat-Completion-Request mit JSON-Antwortformat durch.
    
    Beinhaltet Retry-Logik für temporäre Fehler.
    """
    model = model or config.model_name
    temperature = temperature if temperature is not None else config.temperature
    
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
            if attempt < max_retries - 1 and any(t in text for t in ("429", "rate", "temporar", "timeout", "unavailable")):
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise


def request_next_loop(prev_notes: list[dict], loop_beats: float) -> list[dict]:
    """
    Fordert den nächsten Loop von der KI an.
    
    Fällt bei Fehlern auf Mutation zurück.
    """
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_user_prompt(prev_notes, loop_beats)}
    ]
    try:
        resp = chat_completion_json(messages)
        txt = resp.choices[0].message.content
        return _parse_json_strict(txt, prev_notes)
    except Exception:
        # Fallback: no response_format path
        resp = client.chat.completions.create(
            model=config.model_name,
            temperature=config.temperature,
            messages=messages
        )
        raw = resp.choices[0].message.content
        block = _json_block_from_text(raw)
        if block:
            return _parse_json_strict(block, prev_notes)
        return mutate_seed(prev_notes, max(0.3, config.mutation_amount))

def request_initial_params() -> dict[str, Any]:
    """
    Fordert initiale Parameter von der KI an.
    
    Returns:
        Dictionary mit Parametern (bpm, bars, voices, key, mode, seed)
    """
    prompt = (
        "You are a creative music assistant. "
        "Determine sensible initial parameters for a MIDI sequencer suitable for the following prompt. "
        "Respond ONLY with JSON, e.g.:\n"
        '{ "voices": 4, "bpm": 120, "bars": 2, "key": "C", "mode": "Dorian", "seed": "0,2,4,1,6,2,4,2" }\n'
        f"Prompt:\n{config.style_prompt}\n"
    )
    try:
        resp = chat_completion_json([{"role": "user", "content": prompt}])
        txt = resp.choices[0].message.content
        return json.loads(txt)
    except Exception as e:
        print("[WARN] AI parameter selection failed, using defaults.", e)
        return {}


# ================================
# MIDI
# ================================
def open_midi_out(preferred: Optional[str] = None, interactive: bool = True) -> mido.ports.BaseOutput:
    """
    Öffnet einen MIDI-Output-Port.
    
    Args:
        preferred: Bevorzugter Port-Name
        interactive: Wenn True, zeigt interaktive Auswahl
    
    Returns:
        Geöffneter MIDI-Output-Port
    """
    names = mido.get_output_names()
    
    # Wenn ein bevorzugter Port angegeben ist und existiert, verwende diesen
    if preferred and preferred in names:
        print(f"Using specified port: {preferred}")
        return mido.open_output(preferred)
    
    # Wenn keine Ports verfügbar sind, erstelle virtuellen Port
    if not names:
        fallback_name = "AI Sequencer Out"
        print(f"No MIDI-Out ports found → creating virtual port: '{fallback_name}'")
        print("Note: In Ableton, enable this port as a MIDI input (Preferences → Link/MIDI).")
        return mido.open_output(fallback_name, virtual=True)
    
    # Interaktive Auswahl anzeigen
    if interactive:
        print("\nAvailable MIDI Output Ports:")
        for i, name in enumerate(names, 1):
            print(f"  {i}) {name}")
        
        # Option für virtuellen Port hinzufügen
        virtual_option = len(names) + 1
        print(f"  {virtual_option}) Create new virtual port: 'AI Sequencer Out'")
        
        while True:
            try:
                choice = input(f"\nSelect port (1-{virtual_option}): ").strip()
                if not choice:
                    continue
                    
                choice_num = int(choice)
                
                # Virtueller Port gewählt
                if choice_num == virtual_option:
                    fallback_name = "AI Sequencer Out"
                    print(f"Creating virtual port: '{fallback_name}'")
                    print("Note: In Ableton, enable this port as a MIDI input (Preferences → Link/MIDI).")
                    return mido.open_output(fallback_name, virtual=True)
                
                # Existierender Port gewählt
                if 1 <= choice_num <= len(names):
                    selected_port = names[choice_num - 1]
                    print(f"Opening: {selected_port}")
                    return mido.open_output(selected_port)
                else:
                    print(f"Please enter a number between 1 and {virtual_option}")
                    
            except (ValueError, KeyboardInterrupt):
                print(f"Invalid input. Please enter a number between 1 and {virtual_option}")
            except Exception as e:
                print(f"Error opening port: {e}")
                continue
    
    # Fallback: Automatische Auswahl (wie vorher)
    for key in ("IAC", "IAC Driver", "IAC-Treiber", "Loop", "Virtual"):
        matches = [n for n in names if key.lower() in n.lower()]
        if matches:
            print(f"Auto-selecting port (Match '{key}'): {matches[0]}")
            return mido.open_output(matches[0])
    
    # Letzter Fallback: Ersten verfügbaren Port nehmen
    print(f"Auto-selecting first available port: {names[0]}")
    return mido.open_output(names[0])

def reset_all_notes(midi_out: mido.ports.BaseOutput) -> None:
    """
    Sendet All-Notes-Off und Pitchbend-Reset auf allen Kanälen.
    
    Sendet sowohl CC-Messages als auch explizite Note-Off Messages
    für maximale Kompatibilität mit allen MIDI-Geräten.
    """
    for ch in range(16):
        # Explizite Note-Off für alle 128 MIDI-Noten (maximale Kompatibilität)
        for note in range(128):
            midi_out.send(mido.Message('note_off', note=note, velocity=0, channel=ch))
        
        # CC 123 = All Notes Off
        midi_out.send(mido.Message('control_change', control=ALL_NOTES_OFF_CC, value=0, channel=ch))
        # CC 120 = All Sound Off (stoppt auch ausklingende Noten)
        midi_out.send(mido.Message('control_change', control=ALL_SOUND_OFF_CC, value=0, channel=ch))
        # Pitchbend zurücksetzen
        midi_out.send(mido.Message('pitchwheel', pitch=0, channel=ch))
    
    # MIDI Stop Message senden
    try:
        midi_out.send(mido.Message('stop'))
    except Exception:
        pass
    
    try:
        midi_out.send(mido.Message('reset'))
    except Exception:
        pass

# ================================
# Playback
# ================================
def schedule_events(notes: list[dict], bpm: float) -> list[tuple]:
    """
    Erstellt eine zeitlich sortierte Event-Liste aus Noten.
    
    Args:
        notes: Liste von Noten-Dictionaries
        bpm: Beats per Minute
    
    Returns:
        Sortierte Liste von Events (on, off, pitchbend)
    """
    spb = sec_per_beat(bpm)
    events: list[tuple] = []
    
    for n in notes:
        if n["dur_beats"] <= 0:
            continue
        on_t = n["start_beats"] * spb
        off_t = (n["start_beats"] + n["dur_beats"]) * spb
        ch = n["channel"] - 1
        
        # Pitchbend deaktiviert - war Ursache für "out of tune" Klang
        # if (n["channel"] == 2) and (random.random() < 0.25):
        #     bend_val = random.randint(-200, 200)
        #     events.append(("pitchbend", on_t - 0.03, bend_val, ch))
        #     events.append(("pitchbend", off_t + 0.01, 0, ch))
        
        events.append(("on", on_t, n["midi"], ch, n["velocity"]))
        events.append(("off", off_t, n["midi"], ch, 0))
    
    # Stable sort: by time, then pitchbend -> off -> on
    order = {"pitchbend": 0, "off": 1, "on": 2}
    events.sort(key=lambda e: (e[1], order.get(e[0], 3)))
    return events

def play_loop(
    midi_out: mido.ports.BaseOutput,
    events: list[tuple],
    t0: float,
    send_clock: bool = False,
    bpm: float = 120,
    loop_beats: float = 8
) -> None:
    """
    Spielt einen Loop ab.
    
    Args:
        midi_out: MIDI-Output-Port
        events: Zeitlich sortierte Event-Liste
        t0: Startzeitpunkt (monotonic time)
        send_clock: Ob MIDI-Clock gesendet werden soll
        bpm: Beats per Minute
        loop_beats: Länge des Loops in Beats
    """
    clock_thread: Optional[threading.Thread] = None
    clock_stop_event = threading.Event()  # Sauberer Event statt lokaler Variable
    
    if send_clock:
        midi_out.send(mido.Message('start'))
        
        def clock_worker():
            clock_interval = sec_per_beat(bpm) / MIDI_CLOCK_PPQN
            loop_duration = sec_per_beat(bpm) * loop_beats
            start_time = t0
            next_tick = start_time
            
            while not clock_stop_event.is_set():
                now = time.monotonic()
                if now - start_time >= loop_duration:
                    break
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
        
        jitter = random.uniform(-config.humanize_t, config.humanize_t)
        if jitter > 0:
            time.sleep(jitter)
        
        if kind == "pitchbend":
            bend_val, ch = params
            msg = mido.Message('pitchwheel', pitch=bend_val, channel=ch)
        else:
            note, ch, vel = params
            if kind == 'on':
                # Safety: kill duplicates before re-on
                midi_out.send(mido.Message('note_off', note=note, velocity=0, channel=ch))
            msg = mido.Message('note_on' if kind == 'on' else 'note_off',
                               note=note, velocity=vel, channel=ch)
        midi_out.send(msg)
        i += 1
    
    # Warte bis zum exakten Ende des Loops (verhindert Pausen zwischen Takten)
    loop_duration = sec_per_beat(bpm) * loop_beats
    elapsed = time.monotonic() - t0
    remaining = loop_duration - elapsed
    if remaining > 0:
        time.sleep(remaining)
    
    if send_clock and clock_thread:
        clock_stop_event.set()
        clock_thread.join(timeout=1.0)

# ================================
# Prefetch worker (background)
# ================================
def prefetch_worker(
    q: queue.Queue,
    prev_ref: dict[str, list[dict]],
    loop_beats: float,
    stop_ev: threading.Event,
    refresh_ev: threading.Event,
    lock: threading.Lock
) -> None:
    """
    Hintergrund-Worker für asynchrones Prefetching von KI-Loops.
    
    Args:
        q: Queue für die vorbereiteten Loops
        prev_ref: Referenz auf den aktuellen Loop (wird extern aktualisiert)
        loop_beats: Länge des Loops in Beats
        stop_ev: Event zum Beenden des Workers
        refresh_ev: Event zum Triggern eines neuen Requests
        lock: Threading-Lock für Thread-sichere Zugriffe auf prev_ref
    """
    while not stop_ev.is_set():
        with lock:
            current_notes = prev_ref["notes"][:]  # Kopie für Thread-Sicherheit
        try:
            nxt = request_next_loop(current_notes, loop_beats)
        except Exception as e:
            print("KI-Prefetch Fehler, nutze Mutation:", e)
            nxt = mutate_seed(current_notes, max(0.3, config.mutation_amount))
        
        # Keep only freshest
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        
        try:
            q.put_nowait(nxt)
        except queue.Full:
            pass
        
        # Warte auf Refresh-Signal oder Timeout
        refresh_ev.wait(timeout=0.5)
        refresh_ev.clear()

# ================================
# CLI / main
# ================================
def main() -> None:
    """
    Haupteinstiegspunkt für den AI-Sequencer.
    
    Parst CLI-Argumente, initialisiert die Konfiguration und startet den Jam-Loop.
    """
    global config
    
    # Prompt aus Datei laden
    file_prompt = read_extra_prompt()
    if file_prompt:
        print(f"Prompt aus prompt.txt:\n{file_prompt}\n")
        config.style_prompt = file_prompt
    
    # CLI-Argumente definieren
    p = argparse.ArgumentParser(description="AI → MIDI Jammer")
    p.add_argument("--seed", type=str, default=None, help="Start-Seed: '0,2,4,1,6,2,4,2'")
    p.add_argument("--seed-mode", type=str, choices=["degree", "semitone"], default="degree")
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--mode", type=str, default=None)
    p.add_argument("--voices", type=int, default=None)
    p.add_argument("--bpm", type=float, default=None)
    p.add_argument("--bars", type=int, default=None)
    p.add_argument("--port", type=str, default=config.midi_out_port)
    p.add_argument("--select-device", action="store_true", 
                   help="Zeigt interaktive MIDI-Device-Auswahl")
    p.add_argument("--prefetch", type=int, choices=[0, 1], default=1)
    p.add_argument("--rest-amount", type=float, default=0.15)
    p.add_argument("--send-clock", type=int, choices=[0, 1], default=1)
    p.add_argument("--channel-divs", type=str, default=None,
                   help="Kommagetrennt: z.B. '4,8,16,8' (Viertel/Achtel/Sechzehntel/Achtel)")
    p.add_argument("--mutation", type=float, default=config.mutation_amount)
    p.add_argument("--fraction", type=float, default=config.micro_change_fraction,
                   help="Fraction of notes to potentially mutate (0.0-1.0)")
    p.add_argument("--channel-roots", type=str, default=None,
                   help="Kommagetrennte Root-Noten pro Kanal, z.B. 'C2,G3,E4,C3'")
    p.add_argument("--seed-rng", type=int, default=None, help="Deterministische Random-Seed")
    p.add_argument("--temperature", type=float, default=config.temperature, 
                   help="Temperatur for AI (0.0-2.0)")
    p.add_argument("--arrangement", type=str, choices=["off", "build", "wave", "random", "subtract"], 
                   default="off", help="Arrangement mode: off, build, wave, random, subtract")
    p.add_argument("--arr-loops", type=int, default=4,
                   help="Loops per arrangement stage (default: 4)")
    p.add_argument("--gate-length", type=float, default=0.9,
                   help="Gate length as ratio of step duration (0.1-1.0, default: 0.9)")
    args = p.parse_args()
    
    # Validierung
    if args.bpm is not None and not (20 <= args.bpm <= 300):
        p.error("--bpm must be between 20 and 300")
    if args.bars is not None and args.bars <= 0:
        p.error("--bars must be >= 1")
    if args.rest_amount < 0 or args.rest_amount > 1:
        p.error("--rest-amount must be 0..1")
    if args.fraction < 0 or args.fraction > 1:
        p.error("--fraction must be 0..1")
    if args.seed_rng is not None:
        random.seed(args.seed_rng)
    
    # KI-Parameter holen (immer, aber nur einmal)
    print("[INFO] Hole KI-Parameter...")
    ki_params = request_initial_params()
    
    # Konfiguration aufbauen: KI-Defaults -> CLI-Überschreibungen
    config.bpm = args.bpm if args.bpm is not None else ki_params.get("bpm", config.bpm)
    config.loop_bars = args.bars if args.bars is not None else ki_params.get("bars", config.loop_bars)
    config.voices = clamp(args.voices, 1, 4) if args.voices is not None else ki_params.get("voices", config.voices)
    config.global_key = ki_params.get("key", config.global_key)
    config.global_mode = args.mode if args.mode is not None else ki_params.get("mode", config.global_mode)
    config.mutation_amount = clamp(args.mutation, 0.0, 1.0)
    config.micro_change_fraction = clamp(args.fraction, 0.0, 1.0)
    config.temperature = clamp(args.temperature, 0.0, 2.0)
    config.gate_length = clamp(args.gate_length, 0.1, 1.0)
    
    # Root-Note bestimmen
    if args.root is not None:
        root_str = args.root
    elif "root" in ki_params and re.match(r"^[A-Ga-g][#b]?-?\d+$", str(ki_params["root"])):
        root_str = ki_params["root"]
    else:
        root_str = f"{config.global_key}4"
    
    # Seed-String bestimmen
    seed_str = args.seed if args.seed is not None else ki_params.get("seed", "0,2,4,1,6,2,4,2")
    
    # Skalen-Alias auflösen
    mode_lower = str(config.global_mode).lower()
    config.global_mode = SCALE_ALIASES.get(mode_lower, mode_lower)
    if config.global_mode not in SCALE_STEPS:
        raise ValueError(f"Unbekannte Skala: {config.global_mode}")
    
    # Root-MIDI-Note parsen
    try:
        root_midi = note_to_midi(root_str)
    except Exception as e:
        print(f"[WARN] Fehler beim Parsen von root '{root_str}', nutze 'C4': {e}")
        root_midi = note_to_midi("C4")
    
    loop_beats = beats_per_bar(config.time_sig) * config.loop_bars
    
    # Berechne Skalennoten für Debug-Ausgabe
    scale_steps = SCALE_STEPS[config.global_mode]
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    root_semitone = root_midi % 12
    scale_notes = [note_names[(root_semitone + step) % 12] for step in scale_steps]
    
    # Ausführliche Debug-Ausgabe aller Parameter
    print("\n" + "="*50)
    print("🎹 AI SEQUENCER - Configuration")
    print("="*50)
    print(f"  BPM:           {config.bpm}")
    print(f"  Bars:          {config.loop_bars} ({loop_beats} beats)")
    print(f"  Time Sig:      {config.time_sig[0]}/{config.time_sig[1]}")
    print(f"  Voices:        {config.voices} (Channels: {config.use_channels})")
    print(f"  Root:          {root_str} (MIDI {root_midi})")
    print(f"  Mode:          {config.global_mode}")
    print(f"  Scale Notes:   {' - '.join(scale_notes)}")
    print(f"  Pitch Range:   {config.pitch_range[0]} - {config.pitch_range[1]}")
    print(f"  Velocity:      {config.velocity_range[0]} - {config.velocity_range[1]}")
    print(f"  Mutation:      {config.mutation_amount}")
    print(f"  Fraction:      {config.micro_change_fraction}")
    print(f"  Temperature:   {config.temperature}")
    print(f"  Gate Length:   {config.gate_length} ({int(config.gate_length * 100)}% of step)")
    print(f"  Rest Amount:   {args.rest_amount}")
    print(f"  Seed:          {seed_str}")
    print(f"  Channel Divs:  {args.channel_divs if args.channel_divs else 'Default (0.5 beats)'}")
    print(f"  Arrangement:   {args.arrangement} ({args.arr_loops} loops/stage)")
    print(f"  Prefetch:      {'On' if args.prefetch else 'Off'}")
    print(f"  Send Clock:    {'On' if args.send_clock else 'Off'}")
    print("="*50 + "\n")

    # MIDI out
    interactive_device_selection = args.select_device or (args.port == config.midi_out_port and args.port == DEFAULT_MIDI_OUT_PORT)
    out = open_midi_out(preferred=args.port, interactive=interactive_device_selection)
    reset_all_notes(out)

    # per-channel step durations
    channel_step_durs: Optional[list[float]] = None
    if args.channel_divs:
        divs = [max(1, int(x)) for x in args.channel_divs.split(",")]
        beats_bar = beats_per_bar(config.time_sig)
        channel_step_durs = [beats_bar / d for d in divs]
        # Anzahl der Stimmen an die channel_divs anpassen
        config.voices = min(len(divs), 4)
    
    # per-channel roots (optional)
    channel_roots: Optional[list[int]] = None
    if args.channel_roots:
        channel_roots = [note_to_midi(r.strip()) for r in args.channel_roots.split(",")]
    elif args.root and "," in args.root:
        channel_roots = [note_to_midi(r.strip()) for r in args.root.split(",")]
    if channel_roots:
        while len(channel_roots) < config.voices:
            channel_roots.append(channel_roots[0])
        channel_roots = channel_roots[:config.voices]
    
    # Build seed
    if seed_str:
        seq = parse_seed_list(seed_str)
        if channel_step_durs:
            seqs = [seq for _ in range(len(channel_step_durs))]
            seed = build_seed_from_sequence(
                seqs, root_midi, config.global_mode, seed_mode=args.seed_mode,
                rest_amount=args.rest_amount, channel_step_durs=channel_step_durs,
                loop_beats=loop_beats, channel_roots=channel_roots
            )
        else:
            seed = build_seed_from_sequence(
                seq, root_midi, config.global_mode, seed_mode=args.seed_mode,
                rest_amount=args.rest_amount, loop_beats=loop_beats,
                channel_roots=channel_roots
            )
        prev = seed[:]
    else:
        # Quick chordy seed
        base = root_midi
        chord_int = [0, 3, 7, 10][:config.voices]
        seed: list[dict] = []
        use_channels = config.use_channels
        
        for ch_i in range(len(use_channels)):
            step_dur_ch = channel_step_durs[ch_i] if channel_step_durs else 0.5
            max_beats = loop_beats
            offs = chord_int
            seq_len = len(offs)
            step_count = 0
            steps_list: list[tuple[float, int | None]] = []
            t = 0.0
            
            while t < max_beats:
                val_idx = step_count % seq_len if seq_len > 0 else None
                val = offs[val_idx] if val_idx is not None else None
                is_note = val is not None and (random.random() >= args.rest_amount)
                steps_list.append((t, val if is_note else None))
                t += step_dur_ch
                step_count += 1
            
            note_times = [tt for tt, v in steps_list if v is not None]
            note_vals = [v for tt, v in steps_list if v is not None]
            rng = config.voice_ranges.get(use_channels[ch_i], config.pitch_range)
            
            for i, (tt, val) in enumerate(zip(note_times, note_vals)):
                midi = clamp(base + val, *rng)
                next_t = note_times[i + 1] if i + 1 < len(note_times) else max_beats
                gate_dur = max(0.05, min(step_dur_ch * 0.8, next_t - tt - 0.01))
                seed.append({
                    "start_beats": tt,
                    "dur_beats": gate_dur,
                    "midi": midi,
                    "velocity": random.randint(*config.velocity_range),
                    "channel": use_channels[ch_i % len(use_channels)]
                })
        prev = seed[:]

    scene_loop_count = 0
    total_loop_count = 0  # Für Arrangement-Tracking
    
    # Prefetch mit separatem Refresh-Event
    refresh_ev: Optional[threading.Event] = None
    stop_ev: Optional[threading.Event] = None
    prev_ref: Optional[dict[str, list[dict]]] = None
    q: Optional[queue.Queue] = None
    prev_lock: Optional[threading.Lock] = None
    
    if args.prefetch == 1:
        q = queue.Queue(maxsize=1)
        stop_ev = threading.Event()
        refresh_ev = threading.Event()
        prev_lock = threading.Lock()
        prev_ref = {"notes": prev[:]}
        th = threading.Thread(
            target=prefetch_worker,
            args=(q, prev_ref, loop_beats, stop_ev, refresh_ev, prev_lock),
            daemon=True
        )
        th.start()
    
    # Main jam loop
    try:
        while True:
            next_notes = prev[:]  # Default: repeat
            
            if args.prefetch == 1 and q is not None:
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
                        print("AI request failed – playing identical pattern:", e)
            
            events = schedule_events(next_notes, config.bpm)
            
            # Arrangement: Filtere aktive Kanäle
            if args.arrangement != "off":
                active_channels = get_active_channels(
                    total_loop_count, 
                    config.voices, 
                    args.arrangement,
                    args.arr_loops
                )
                filtered_notes = filter_notes_by_channels(next_notes, active_channels)
                events = schedule_events(filtered_notes, config.bpm)
                
                # Zeige aktive Kanäle an (nur bei Änderung)
                if total_loop_count % args.arr_loops == 0:
                    # Debug: Zeige Noten pro Kanal vor und nach Filter
                    channels_before = {}
                    for n in next_notes:
                        ch = n.get("channel")
                        channels_before[ch] = channels_before.get(ch, 0) + 1
                    channels_after = {}
                    for n in filtered_notes:
                        ch = n.get("channel")
                        channels_after[ch] = channels_after.get(ch, 0) + 1
                    print(f"[ARR] Loop {total_loop_count}: Active channels = {active_channels}")
                    print(f"      Notes before filter: {channels_before}")
                    print(f"      Notes after filter:  {channels_after}")
            
            t0 = time.monotonic()
            play_loop(out, events, t0, send_clock=bool(args.send_clock),
                      bpm=config.bpm, loop_beats=loop_beats)
            
            prev = next_notes
            scene_loop_count += 1
            total_loop_count += 1
            
            if scene_loop_count >= config.scene_hold_loops:
                prev = mutate_seed(prev, config.mutation_amount)
                scene_loop_count = 0
            
            if args.prefetch == 1 and prev_ref is not None and refresh_ev is not None and prev_lock is not None:
                with prev_lock:
                    prev_ref["notes"] = prev
                refresh_ev.set()  # Signal zum Worker für neuen Request
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted – closing all gates.")
        if stop_ev is not None:
            stop_ev.set()
    
    finally:
        if stop_ev is not None:
            stop_ev.set()
        reset_all_notes(out)
        print("[INFO] All MIDI gates have been closed.")


if __name__ == "__main__":
    # mido.set_backend('mido.backends.rtmidi')  # uncomment if needed
    main()
