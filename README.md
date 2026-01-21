# AI Evolving Jam Prompted

This project is an AI-based MIDI sequencer that automatically generates and evolves musical patterns using OpenAI. The patterns are output via MIDI and can be further processed in Ableton or hardware sequencers.

## Requirements

- Python 3.8+
- Python packages: `openai`, `mido`, `python-rtmidi`
- An OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- Optional: Virtual MIDI port (e.g., IAC Driver on macOS)

## Installation

```bash
pip install openai mido python-rtmidi
```

## Usage

1. **Adjust the prompt:**  
   Write your musical idea in the file `prompt.txt` (e.g., style, rhythm, desired scale).

2. **Start the script:**  
   Change to the project directory and start the script:
   ```bash
   python3 ai_evolving_jam_prompted.py
   ```

3. **Parameters (optional):**  
   You can control the behavior via CLI parameters, e.g.:
   ```bash
   python3 ai_evolving_jam_prompted.py --voices 3 --bpm 120 --bars 4 --mode Dorian --root C4
   ```
   Without parameters, the AI determines suitable initial values based on the prompt.

### Parameter Description

| Parameter         | Type     | Description                                                                                 | Example                         |
|-------------------|----------|---------------------------------------------------------------------------------------------|----------------------------------|
| `--voices`        | int      | Number of voices/channels (1-4)                                                             | `--voices 3`                    |
| `--bpm`           | float    | Tempo in beats per minute                                                                   | `--bpm 120`                     |
| `--bars`          | int      | Number of bars per loop                                                                     | `--bars 4`                      |
| `--mode`          | str      | Scale/mode (e.g., Dorian, Phrygian, Major, Minor)                                           | `--mode Dorian`                 |
| `--root`          | str      | Root note (e.g., C4, D#3)                                                                   | `--root C4`                     |
| `--seed`          | str      | Initial seed as a sequence of numbers (e.g., note intervals)                                | `--seed "0,2,4,1,6,2,4,2"`      |
| `--seed-mode`     | str      | Interpretation of the seed: `degree` (scale degree) or `semitone` (semitone)                | `--seed-mode degree`            |
| `--port`          | str      | Name of the MIDI output port                                                                | `--port "IAC Driver Bus 1"`     |
| `--select-device` | flag     | Shows interactive MIDI device selection menu                                                | `--select-device`               |
| `--prefetch`      | int      | 1=enable AI prefetch, 0=disable                                                             | `--prefetch 1`                  |
| `--rest-amount`   | float    | Probability of rests (0.0-1.0)                                                              | `--rest-amount 0.15`            |
| `--send-clock`    | int      | Send MIDI clock/start: 1=on, 0=off                                                          | `--send-clock 1`                |
| `--channel-divs`  | str      | Comma-separated note divisions per channel (e.g., '4,8,16,8' for quarter, eighth, etc.)     | `--channel-divs "4,8,16,8"`     |
| `--mutation`      | float    | Mutation strength for pattern evolution (0.0-1.0)                                           | `--mutation 0.35`               |
| `--fraction`      | float    | Fraction of notes to potentially mutate per loop (0.0-1.0)                                  | `--fraction 0.3`                |
| `--channel-roots` | str      | Comma-separated root notes per channel                                                      | `--channel-roots "C2,G3,E4,C3"` |
| `--seed-rng`      | int      | Deterministic random seed for reproducibility                                               | `--seed-rng 42`                 |
| `--temperature`   | float    | AI temperature for creativity (0.0-2.0)                                                     | `--temperature 0.6`             |
| `--arrangement`   | str      | Arrangement mode: off, build, wave, random, subtract                                        | `--arrangement build`           |
| `--arr-loops`     | int      | Number of loops per arrangement stage                                                       | `--arr-loops 4`                 |
| `--gate-length`   | float    | Gate length as ratio of step duration (0.1-1.0)                                             | `--gate-length 0.9`             |

## Note on Clock Divisions per Channel

You can set the clock divisions for each channel directly via the `--channel-divs` parameter.  
Example:  
```bash
python3 ai_evolving_jam_prompted.py --channel-divs "4,8,16,8"
```
This means:
- Channel 1 plays quarter notes (4)
- Channel 2 plays eighth notes (8)
- Channel 3 plays sixteenth notes (16)
- Channel 4 plays eighth notes (8)

**You do not need to specify this in the prompt itself.**  
The divisions are controlled directly via the parameter and are considered in the pattern.

4. **MIDI Setup:**  
   The script automatically opens a suitable MIDI output port.  
   
   **Interactive Device Selection:**
   - Use `--select-device` to get an interactive menu showing all available MIDI ports
   - You can then choose from:
     - Available hardware MIDI devices
     - Virtual MIDI ports (e.g., IAC Driver on macOS)
     - Option to create a new virtual port
   
   **Automatic Selection:**
   - Without `--select-device`, the script will automatically select a suitable port
   - First tries the port specified with `--port` parameter
   - Then looks for common virtual ports (IAC Driver, Loop, Virtual)
   - Finally creates a new virtual port if none found
   
   In Ableton or your DAW, activate the chosen port as a MIDI input.

5. **AI Generation:**  
   The AI continuously generates new patterns that subtly evolve.  
   The prompt from `prompt.txt` influences the musical direction.

6. **Stop:**  
   Press `Ctrl+C` to stop the script and close all MIDI gates.

## Notes

- The `--rest-amount` parameter controls how likely it is that rests are generated in the pattern.
- The file `prompt.txt` is the creative core. The more precise and musical the text, the better the result.
- Scales like "Major" and "Minor" are automatically mapped to "Ionian" and "Aeolian".
- The script is intended for experimental and generative music.

## Example for prompt.txt

```plaintext
Create a 16-step pattern in the style of Aphex Twin: irregular rhythms, sudden syncopations, and abrupt accents.
Use unusual scales (e.g., Phrygian or Dorian), include chromatic approaches and occasional unexpected jumps of more than an octave.
The bass line should sound mechanical and repetitive, while the melody remains edgy and unpredictable.
Add a 20% probability for very short ghost notes.
Keep the overall character dark, mechanical, but driving. And evolve the notes over time.
```

## Error Handling

- If the AI returns an invalid value for the root note, it automatically falls back to "C4".
- On interruption, all notes and pitch bends are reset.

---

Have fun with generative jamming!
