#!/bin/bash
# AI Sequencer Startup Script
# Alle Parameter können hier angepasst werden

# ============================================
# KONFIGURATION
# ============================================

# Tempo & Timing
BPM=90
BARS=2

# Voices & Scale
VOICES=4
ROOT="G#4"
MODE="Minor"

# Seed (Skalengrade als Startwerte, 1-basiert: 1=Wurzel, 3=Terz, 5=Quinte)
SEED="1,5,1,6,1,7,6,5"
SEED_MODE="degree"  # degree oder semitone

# Evolution
MUTATION=0.5
FRACTION=0.3
TEMPERATURE=0.4

# Rhythm
REST_AMOUNT=0
CHANNEL_DIVS="4,8,16,2"  # Viertel, Achtel, Achtel, Sechzehntel
GATE_LENGTH=0.50  # Gate-Länge (0.1-1.0, 1.0 = legato)

# Optional: Verschiedene Root-Noten pro Kanal
# CHANNEL_ROOTS="E2,E3,B3,E4"

# Playback
PREFETCH=1
SEND_CLOCK=1

# Arrangement (off, build, wave, random, subtract)
ARRANGEMENT="off"
ARR_LOOPS=4  # Loops pro Stage

# Für reproduzierbare Ergebnisse (optional)
# SEED_RNG=42

# ============================================
# SCRIPT STARTEN
# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"
SCRIPT="$SCRIPT_DIR/ai_evolving_jam_prompted.py"

# Baue Kommandozeile
CMD="$PYTHON $SCRIPT"
CMD="$CMD --bpm $BPM"
CMD="$CMD --bars $BARS"
CMD="$CMD --voices $VOICES"
CMD="$CMD --root $ROOT"
CMD="$CMD --mode $MODE"
CMD="$CMD --seed \"$SEED\""
CMD="$CMD --seed-mode $SEED_MODE"
CMD="$CMD --mutation $MUTATION"
CMD="$CMD --fraction $FRACTION"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --rest-amount $REST_AMOUNT"
CMD="$CMD --gate-length $GATE_LENGTH"
CMD="$CMD --prefetch $PREFETCH"
CMD="$CMD --send-clock $SEND_CLOCK"

# Optionale Parameter
if [ -n "$CHANNEL_DIVS" ]; then
    CMD="$CMD --channel-divs \"$CHANNEL_DIVS\""
fi

if [ -n "$CHANNEL_ROOTS" ]; then
    CMD="$CMD --channel-roots \"$CHANNEL_ROOTS\""
fi

if [ -n "$SEED_RNG" ]; then
    CMD="$CMD --seed-rng $SEED_RNG"
fi

if [ -n "$ARRANGEMENT" ]; then
    CMD="$CMD --arrangement $ARRANGEMENT"
    CMD="$CMD --arr-loops $ARR_LOOPS"
fi

# Interaktive Device-Auswahl
CMD="$CMD --select-device"

echo "Starting AI Sequencer..."
echo "Command: $CMD"
echo ""

eval $CMD
