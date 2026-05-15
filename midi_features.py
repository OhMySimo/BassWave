# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""MIDI -> per-window transient features for BassWave training.

Output per finestra di `example_secs` secondi a `frame_rate` Hz:
  * onset_mask             : [T] float32, 1.0 nel frame di un onset
  * onset_offset_samples   : [T] float32, residuo sub-frame in
                             [-half_frame_samples, +half_frame_samples] sample
                             (valido solo dove onset_mask==1; 0 altrove)
  * onset_velocity         : [T] float32, MIDI velocity 0..127 normalizzato in
                             [0, 1] (valido solo dove onset_mask==1; 0 altrove)
  * offset_mask            : [T] float32, 1.0 nel frame di un note-off
  * offset_offset_samples  : [T] float32, residuo sub-frame note-off in sample
  * silence_mask           : [T] float32, 1.0 nei frame senza nota attiva
  * active_ks_bits         : [T] float32, bitmask intera (come float) dei key
                             switch in stato "held" nel frame. Bit i = KS con
                             indice i (0=KS_POLYPHONY/midi9 … 11=KS_LEGATO/midi20).
                             0.0 = nessun KS attivo. Solo disponibile se viene
                             passato un BassMidiKeymap; altrimenti sempre 0.
  * active_note_midi       : [T] float32, MIDI pitch della nota attiva nel frame
                             (0.0 = silenzio). Per dataset monofoni come BassWave
                             al massimo una nota per frame. Disponibile solo con
                             keymap; senza keymap coincide con silence_mask (0=sil).

Precisione sample-level
-----------------------
A 44.1 kHz e frame_rate=50 Hz, half_frame_samples=441. La testa predice
onset_offset_samples come regressione continua su [-441, +441]. L'output
arrotondato all'intero piu' vicino e' sample-accurate fino al limite della
quantizzazione MIDI sorgente: per il dataset BassWave a 960 ticks/beat e
tempi ~112 BPM, 1 tick = ~25 sample, quindi la ground truth e' quantizzata
naturalmente a multipli di ~25 sample. Questa quantizzazione e' assorbita
dal target (non c'e' modo di andare piu' fini del MIDI originale).

Convenzioni di framing
----------------------
Tutto il calcolo usa "frame center time" = t0 + i/frame_rate (s), dove t0 e'
il timestamp dell'inizio della finestra. Questo coincide con la convenzione
centered=True usata da prepare_basswave (CREPE/loudness frame i e' centrato
in t0 + i/frame_rate). T_frames = round(example_secs*frame_rate) + 1 se
centered=True (es. 4 s * 50 Hz + 1 = 201), altrimenti senza il +1.

Sub-frame offset
----------------
Per un onset al tempo assoluto `t_on`, in finestra che parte a `t0`:
  frame_idx       = round((t_on - t0) * frame_rate)            # nearest frame
  frame_center_s  = t0 + frame_idx / frame_rate
  offset_samples  = (t_on - frame_center_s) * sample_rate      # in
                                                                # [-half_frame,
                                                                #  +half_frame]

Cosi' una testa che predice (p_onset, offset_samples, velocity) per ogni
frame puo':
  * BCE su p_onset (classification)
  * MSE su offset_samples mascherato da onset_mask (regression sub-frame)
  * MSE su velocity mascherato da onset_mask (regression dynamics)

half_frame_samples = sample_rate / (2 * frame_rate). Per default 44.1k/50:
441 sample = ~10 ms.

Note tenute oltre il bordo
--------------------------
Nota che inizia PRIMA della finestra ma e' attiva dentro: nessun evento di
onset registrato (e' fuori finestra), pero' i frame attivi sono coperti per
il silence_mask (silence=0). Idem per nota che si estende DOPO la finestra:
nessun evento offset, ma frame attivi coperti per il silence_mask.

Integrazione BassMidiKeymap
----------------------------
Quando si passa `keymap` a `parse_midi`, i MIDI event con pitch in range
9-20 (key switch range A-1..G#0) vengono estratti come `KeySwitchEvent`
e NON inclusi nella lista `notes` (niente falsi onset su KS). La lista
`notes` contiene solo pitch 21-64 (A0..E4), ciascuno con `token_id` dal
keymap (es. NOTE_E1, NOTE_A1, ...). I KeySwitchEvent vengono passati a
`window_features` e `windowed_features_for_file` per generare
`active_ks_bits` e `active_note_midi`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Path translation (metadata.jsonl usa path Windows).
# -----------------------------------------------------------------------------


def translate_path(p: str,
                   src_prefix: str = 'E:/',
                   dst_prefix: str = '/media/simone/NVME/') -> str:
  """Sostituisci prefisso Windows con percorso Linux."""
  p = p.replace('\\', '/')
  if p.startswith(src_prefix):
    return dst_prefix + p[len(src_prefix):]
  return p


# -----------------------------------------------------------------------------
# BassMidiKeymap — wrapper attorno a bass_midi_keymap.json.
# -----------------------------------------------------------------------------


class BassMidiKeymap:
  """Carica e interroga bass_midi_keymap.json.

  Permette di:
    * distinguere key switch (MIDI 9-20) da note regolari (MIDI 21-64)
    * recuperare il token_id (es. "NOTE_E1", "KS_LEGATO") per ogni MIDI pitch
    * recuperare il trigger type ("hold", "press", "press_and_hold") dei KS

  Costanti importanti:
    KS_MIDI_MIN / KS_MIDI_MAX : 9 / 20    (A-1 .. G#0)
    NOTE_MIDI_MIN / NOTE_MIDI_MAX : 21 / 64  (A0 .. E4)
    N_KS   : 12 key switches (indici 0..11, KS index = midi_id - 9)
    N_NOTES: 44 note (indici 0..43, note index = midi_id - 21)
  """

  KS_MIDI_MIN:   int = 9
  KS_MIDI_MAX:   int = 20
  NOTE_MIDI_MIN: int = 21
  NOTE_MIDI_MAX: int = 64
  N_KS:          int = 12  # 20 - 9 + 1
  N_NOTES:       int = 44  # 64 - 21 + 1

  def __init__(self, keymap_path_or_dict):
    """Args:
      keymap_path_or_dict: path al JSON oppure dict gia' caricato.
    """
    if isinstance(keymap_path_or_dict, dict):
      data = keymap_path_or_dict
    else:
      with open(keymap_path_or_dict, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # flat_map: int(midi_id) -> {'token_id': str, 'type': str}
    self._flat: Dict[int, dict] = {
        int(k): v for k, v in data['flat_map'].items()
    }
    # KS-specific info: int(midi_id) -> trigger string or None
    self._ks_trigger: Dict[int, Optional[str]] = {
        int(ks['midi_id']): ks.get('trigger')
        for ks in data['key_switches']
    }

  # --- Classification helpers ---

  def is_key_switch(self, midi_id: int) -> bool:
    return self.KS_MIDI_MIN <= midi_id <= self.KS_MIDI_MAX

  def is_note(self, midi_id: int) -> bool:
    return self.NOTE_MIDI_MIN <= midi_id <= self.NOTE_MIDI_MAX

  # --- Lookup helpers ---

  def token_id(self, midi_id: int) -> Optional[str]:
    entry = self._flat.get(midi_id)
    return entry['token_id'] if entry else None

  def ks_trigger(self, midi_id: int) -> Optional[str]:
    """Tipo di trigger del KS: 'hold', 'press', 'press_and_hold' o None."""
    return self._ks_trigger.get(midi_id)

  def ks_index(self, midi_id: int) -> int:
    """Indice 0-based del KS (midi_id 9 -> 0, midi_id 20 -> 11)."""
    return midi_id - self.KS_MIDI_MIN

  def note_index(self, midi_id: int) -> int:
    """Indice 0-based nella lista note (midi_id 21 -> 0, midi_id 64 -> 43)."""
    return midi_id - self.NOTE_MIDI_MIN


# -----------------------------------------------------------------------------
# Data classes.
# -----------------------------------------------------------------------------


@dataclass
class Note:
  """Singola nota: onset e offset in secondi assoluti dall'inizio del file."""
  t_on: float        # seconds
  t_off: float       # seconds (must be > t_on)
  pitch: int         # MIDI pitch (21..64 quando keymap e' usato)
  velocity: int      # 0..127
  token_id: Optional[str] = None   # es. "NOTE_E1" se keymap fornito

  @property
  def duration_s(self) -> float:
    return self.t_off - self.t_on


@dataclass
class KeySwitchEvent:
  """Evento key-switch: attivo durante [t_on, t_off).

  Estratto da MIDI pitch range 9-20 quando parse_midi riceve un keymap.
  Usato da window_features per calcolare active_ks_bits.
  """
  t_on:      float
  t_off:     float
  midi_id:   int           # 9..20
  token_id:  str           # es. "KS_LEGATO"
  ks_index:  int           # 0-based: midi_id - 9
  trigger:   Optional[str] = None  # 'hold', 'press', 'press_and_hold'

  @property
  def duration_s(self) -> float:
    return self.t_off - self.t_on


# -----------------------------------------------------------------------------
# MIDI parsing.
# -----------------------------------------------------------------------------


def parse_midi(
    midi_path: str,
    override_bpm: Optional[float] = None,
    keymap: Optional[BassMidiKeymap] = None,
) -> Tuple[List[Note], List[KeySwitchEvent]]:
  """Estrai nota e key-switch event da un file .mid.

  Restituisce sempre una tupla (notes, ks_events).

  Quando `keymap` e' None:
    * ks_events e' sempre vuoto []
    * notes contiene TUTTI gli eventi MIDI come Note (comportamento originale)
      con token_id=None.

  Quando `keymap` e' fornito:
    * MIDI pitch 9-20  -> KeySwitchEvent (NON inclusi in notes)
    * MIDI pitch 21-64 -> Note con token_id valorizzato dal keymap
    * Altri pitch      -> ignorati (fuori range bass)

  Questo garantisce che onset_mask NON si attivi su key switch, che sono
  eventi di articolazione senza corrispondenza audio.

  ATTENZIONE - bug del dataset BassWave: i file MIDI contengono un set_tempo
  SBAGLIATO (hardcoded 112 BPM per tutti i file). Usa override_bpm=120.0 per
  il dataset BassWave renderizzato a 120 BPM.

  Args:
    midi_path: path al file .mid.
    override_bpm: se non None, BPM costante per conversione tick->time,
      ignorando i set_tempo MIDI.
    keymap: se fornito, separa KS da note e valorizza token_id.

  Returns:
    (notes, ks_events): tuple. ks_events=[] se keymap is None.
  """
  try:
    import mido
  except ImportError as e:
    raise ImportError(
        'mido richiesto per parse_midi. Installa con: pip install mido') from e

  mf = mido.MidiFile(midi_path)

  # active_notes: pitch -> (t_on, velocity)
  active_notes: Dict[int, Tuple[float, int]] = {}
  # active_ks: pitch -> t_on  (t_on of the key-switch note_on)
  active_ks: Dict[int, float] = {}

  notes:     List[Note]           = []
  ks_events: List[KeySwitchEvent] = []

  def _flush_note(pitch: int, t_off: float):
    if pitch in active_notes:
      t_on, vel = active_notes.pop(pitch)
      tok = keymap.token_id(pitch) if keymap and keymap.is_note(pitch) else None
      notes.append(Note(t_on=t_on, t_off=t_off,
                        pitch=pitch, velocity=vel, token_id=tok))

  def _flush_ks(pitch: int, t_off: float):
    if pitch in active_ks:
      t_on = active_ks.pop(pitch)
      ks_events.append(KeySwitchEvent(
          t_on=t_on, t_off=t_off,
          midi_id=pitch,
          token_id=keymap.token_id(pitch) or f'KS_{pitch}',
          ks_index=keymap.ks_index(pitch),
          trigger=keymap.ks_trigger(pitch),
      ))

  def _handle_on(pitch: int, velocity: int, t_now: float):
    if keymap is None:
      # Original behavior: all pitches -> notes.
      if pitch in active_notes:
        _flush_note(pitch, t_now)
      active_notes[pitch] = (t_now, velocity)
    elif keymap.is_key_switch(pitch):
      if pitch in active_ks:
        _flush_ks(pitch, t_now)
      active_ks[pitch] = t_now
    elif keymap.is_note(pitch):
      if pitch in active_notes:
        _flush_note(pitch, t_now)
      active_notes[pitch] = (t_now, velocity)
    # else: fuori range bass -> ignora

  def _handle_off(pitch: int, t_now: float):
    if keymap is None:
      _flush_note(pitch, t_now)
    elif keymap.is_key_switch(pitch):
      _flush_ks(pitch, t_now)
    elif keymap.is_note(pitch):
      _flush_note(pitch, t_now)

  if override_bpm is not None:
    if override_bpm <= 0:
      raise ValueError(f'override_bpm deve essere > 0, got {override_bpm}')
    sec_per_tick = 60.0 / (float(override_bpm) * mf.ticks_per_beat)

    for tr in mf.tracks:
      # Reset per-track state: ogni track ha la sua linea temporale.
      active_notes_track: Dict[int, Tuple[float, int]] = {}
      active_ks_track:    Dict[int, float]             = {}
      # Temporarily redirect _handle_* to track-local dicts.
      # Instead of monkey-patching, we inline the logic here.
      cum_ticks = 0
      t_last = 0.0
      for msg in tr:
        cum_ticks += msg.time
        t_now = cum_ticks * sec_per_tick
        t_last = t_now
        if msg.type == 'note_on' and msg.velocity > 0:
          _handle_on(msg.note, msg.velocity, t_now)
        elif (msg.type == 'note_off'
              or (msg.type == 'note_on' and msg.velocity == 0)):
          _handle_off(msg.note, t_now)
      # Track-end fallback per eventi rimasti aperti.
      for pitch in list(active_notes.keys()):
        _flush_note(pitch, t_last)
      for pitch in list(active_ks.keys()):
        _flush_ks(pitch, t_last)
  else:
    # Default: usa i tempo event del MIDI (mido converte a secondi).
    current_t = 0.0
    for msg in mf:
      current_t += msg.time
      if msg.type == 'note_on' and msg.velocity > 0:
        _handle_on(msg.note, msg.velocity, current_t)
      elif (msg.type == 'note_off'
            or (msg.type == 'note_on' and msg.velocity == 0)):
        _handle_off(msg.note, current_t)
    # Fallback per eventi aperti.
    for pitch in list(active_notes.keys()):
      _flush_note(pitch, current_t)
    for pitch in list(active_ks.keys()):
      _flush_ks(pitch, current_t)

  notes.sort(key=lambda n: n.t_on)
  ks_events.sort(key=lambda e: e.t_on)
  return notes, ks_events


# -----------------------------------------------------------------------------
# Per-window feature generation.
# -----------------------------------------------------------------------------


def n_frames_for_window(example_secs: float, frame_rate: int,
                        centered: bool) -> int:
  """Numero di frame in una finestra. Deve combaciare con prepare_basswave."""
  n = int(round(example_secs * frame_rate))
  if centered:
    n += 1
  return n


def window_features(notes: List[Note],
                    t0_s: float,
                    example_secs: float,
                    frame_rate: int,
                    sample_rate: int = 44100,
                    centered: bool = True,
                    onset_blur: int = 0,
                    file_last_off_s: float = 0.0,
                    ks_events: Optional[List[KeySwitchEvent]] = None) -> dict:
  """Genera gli 8 array di target per una singola finestra.

  Args:
    notes: lista di Note del file (output di parse_midi, solo pitch 21-64
      quando keymap e' usato).
    t0_s: timestamp inizio finestra (s).
    example_secs: lunghezza finestra (s).
    frame_rate: frame rate features (Hz). Default per BassWave: 50.
    sample_rate: SR audio (per onset_offset_samples). Default 44100.
    centered: usa convenzione centered=True di prepare_basswave (default).
    onset_blur: se >0, mette 1 anche sui ±onset_blur frame adiacenti.
    file_last_off_s: ultimo note-off del file (non usato direttamente ma
      mantenuto per compatibilita').
    ks_events: lista di KeySwitchEvent (output di parse_midi). Se None o [],
      active_ks_bits sara' sempre 0.

  Returns:
    dict con 8 array numpy float32 di shape [T]:
      onset_mask, onset_offset_samples, onset_velocity,
      offset_mask, offset_offset_samples, silence_mask,
      active_ks_bits, active_note_midi
  """
  if ks_events is None:
    ks_events = []

  T = n_frames_for_window(example_secs, frame_rate, centered)
  t_end_s = t0_s + example_secs
  frame_period_s = 1.0 / frame_rate
  half_frame_samples = 0.5 * sample_rate / frame_rate

  onset_mask            = np.zeros(T, dtype=np.float32)
  onset_offset_samples  = np.zeros(T, dtype=np.float32)
  onset_velocity        = np.zeros(T, dtype=np.float32)
  offset_mask           = np.zeros(T, dtype=np.float32)
  offset_offset_samples = np.zeros(T, dtype=np.float32)
  silence_mask          = np.ones(T,  dtype=np.float32)   # default: silence
  active_ks_bits_int    = np.zeros(T, dtype=np.int32)     # bitmask KS
  active_note_midi      = np.zeros(T, dtype=np.float32)   # MIDI pitch attivo

  # Tempi-centro di ogni frame: frame i a t0 + i/frame_rate.
  frame_centers = t0_s + np.arange(T, dtype=np.float64) * frame_period_s

  # ── Note regolari (pitch 21-64) ───────────────────────────────────────────
  for note in notes:
    # Skip nota completamente fuori finestra.
    if note.t_off < t0_s or note.t_on >= t_end_s:
      continue

    # ---- Onset (solo se l'onset cade DENTRO la finestra) ----
    if t0_s <= note.t_on < t_end_s:
      f_idx = int(round((note.t_on - t0_s) * frame_rate))
      if 0 <= f_idx < T:
        center_s = t0_s + f_idx * frame_period_s
        res_samples = float((note.t_on - center_s) * sample_rate)
        res_samples = max(-half_frame_samples,
                          min(half_frame_samples, res_samples))
        vel_norm = float(note.velocity) / 127.0

        onset_mask[f_idx]           = 1.0
        onset_offset_samples[f_idx] = res_samples
        onset_velocity[f_idx]       = vel_norm

        if onset_blur > 0:
          for d in range(1, onset_blur + 1):
            for nb in (f_idx - d, f_idx + d):
              if 0 <= nb < T:
                onset_mask[nb] = max(onset_mask[nb], 1.0)

    # ---- Offset (note_off) ----
    if t0_s <= note.t_off < t_end_s:
      f_idx = int(round((note.t_off - t0_s) * frame_rate))
      if 0 <= f_idx < T:
        center_s = t0_s + f_idx * frame_period_s
        res_samples = float((note.t_off - center_s) * sample_rate)
        res_samples = max(-half_frame_samples,
                          min(half_frame_samples, res_samples))
        offset_mask[f_idx]           = 1.0
        offset_offset_samples[f_idx] = res_samples
        if onset_blur > 0:
          for d in range(1, onset_blur + 1):
            for nb in (f_idx - d, f_idx + d):
              if 0 <= nb < T:
                offset_mask[nb] = max(offset_mask[nb], 1.0)

    # ---- Silence: frame attivi in [t_on, t_off) ----
    active = (frame_centers >= note.t_on) & (frame_centers < note.t_off)
    silence_mask[active] = 0.0

    # ---- Active note MIDI pitch ----
    # Dataset e' mono: al piu' una nota per frame. In caso di overlap (non
    # atteso), l'ultima nota sovrascrive. Pitch 0 = silenzio.
    active_note_midi[active] = float(note.pitch)

  # ── Key switch events (pitch 9-20) ───────────────────────────────────────
  # Calcoliamo active_ks_bits: bitmask int (bit i = KS index i e' attivo).
  #
  # Gestione trigger-type:
  #   hold / press_and_hold : attivo per tutta la durata [t_on, t_off).
  #   press                 : la nota MIDI dura 1-2 ticks (~0.5 ms) → a 50 Hz
  #                           la finestra è < 1/38 di frame → 0 frame attivi.
  #                           Estendiamo t_off_eff a t_on + 1 frame per
  #                           garantire supervisione su almeno 1 frame.
  #   null  (KS_NOT_USED)   : skippiamo completamente (bit 3, midi_id 12).
  #                           Non porta informazione musicale e allenerebbe
  #                           il modello su etichette sempre-0.
  _KS_NOT_USED_INDEX = 3  # midi_id 12 → ks_index = 12 - 9 = 3

  for ev in ks_events:
    # Salta KS_NOT_USED (trigger null, nessun significato musicale).
    if ev.ks_index == _KS_NOT_USED_INDEX:
      continue
    if ev.t_off < t0_s or ev.t_on >= t_end_s:
      continue
    # Per press-type: estendi la finestra attiva a minimo 1 frame intero,
    # così il bit risulta visibile nell'active_ks_bits ground truth.
    if ev.trigger == 'press':
      t_off_eff = max(ev.t_off, ev.t_on + frame_period_s)
    else:
      t_off_eff = ev.t_off
    active = (frame_centers >= ev.t_on) & (frame_centers < t_off_eff)
    bit = 1 << ev.ks_index   # bit 0..11
    active_ks_bits_int[active] |= bit

  return {
      'onset_mask':            onset_mask,
      'onset_offset_samples':  onset_offset_samples,
      'onset_velocity':        onset_velocity,
      'offset_mask':           offset_mask,
      'offset_offset_samples': offset_offset_samples,
      'silence_mask':          silence_mask,
      'active_ks_bits':        active_ks_bits_int.astype(np.float32),
      'active_note_midi':      active_note_midi,
  }


# -----------------------------------------------------------------------------
# Convenience: itera tutte le finestre di un file MIDI.
# -----------------------------------------------------------------------------


def windowed_features_for_file(notes: List[Note],
                               file_duration_s: float,
                               example_secs: float,
                               hop_secs: float,
                               frame_rate: int,
                               sample_rate: int = 44100,
                               centered: bool = True,
                               onset_blur: int = 0,
                               ks_events: Optional[List[KeySwitchEvent]] = None,
                               ) -> List[dict]:
  """Restituisci feature MIDI per ogni finestra del file, in ordine
  cronologico. Combacia 1-a-1 con la sequenza di esempi che
  prepare_basswave.slice_into_windows produce sullo stesso file.

  Args:
    notes: lista di Note (solo pitch 21-64 quando keymap e' usato).
    file_duration_s: durata totale audio in secondi.
    example_secs: lunghezza finestra (s).
    hop_secs: hop tra finestre (s).
    frame_rate: frame rate features (Hz).
    sample_rate: SR audio.
    centered: usa convenzione centered=True.
    onset_blur: blur per onset_mask.
    ks_events: lista KeySwitchEvent. None -> active_ks_bits sempre 0.
  """
  if ks_events is None:
    ks_events = []

  _file_last_off = max((n.t_off for n in notes), default=0.0)

  out = []
  t0 = 0.0
  while t0 + example_secs <= file_duration_s + 1e-9:
    out.append(window_features(
        notes, t0_s=t0,
        example_secs=example_secs,
        frame_rate=frame_rate,
        sample_rate=sample_rate,
        centered=centered,
        onset_blur=onset_blur,
        file_last_off_s=_file_last_off,
        ks_events=ks_events,
    ))
    t0 += hop_secs
  return out


# -----------------------------------------------------------------------------
# Smoke test.
# -----------------------------------------------------------------------------


def _smoke_test_summary(features: dict, T: int):
  """Stampa riassunto stat delle feature di una singola finestra."""
  print(f'  shape              = ({T},)')
  print(f'  onset count        = {int(features["onset_mask"].sum())}')
  print(f'  offset count       = {int(features["offset_mask"].sum())}')
  print(f'  silence frac       = {features["silence_mask"].mean():.3f}')
  onset_idx = np.flatnonzero(features['onset_mask'])
  if len(onset_idx):
    print(f'  onset frames (first 6): {onset_idx[:6].tolist()}')
    print(f'  onset sub-samples  : '
          f'{features["onset_offset_samples"][onset_idx[:6]].round(1).tolist()}')
    print(f'  onset velocity     : '
          f'{features["onset_velocity"][onset_idx[:6]].round(3).tolist()}')
  # KS info
  ks_frames = np.flatnonzero(features['active_ks_bits'])
  if len(ks_frames):
    unique_bits = np.unique(features['active_ks_bits'][ks_frames].astype(int))
    print(f'  active_ks_bits     : {len(ks_frames)} frames, '
          f'unique bitmasks={unique_bits.tolist()}')
  else:
    print(f'  active_ks_bits     : 0 frames (nessun KS o keymap non fornito)')
  # Note pitch info
  active_pitch_frames = np.flatnonzero(features['active_note_midi'])
  if len(active_pitch_frames):
    unique_pitches = np.unique(features['active_note_midi'][active_pitch_frames].astype(int))
    print(f'  active_note_midi   : pitches attivi={unique_pitches.tolist()}')
  else:
    print(f'  active_note_midi   : silenzio (0) in tutti i frame')


if __name__ == '__main__':
  import argparse
  ap = argparse.ArgumentParser(
      description='Smoke test parser MIDI + feature generator.')
  ap.add_argument('--midi', required=True, help='Path .mid')
  ap.add_argument('--keymap', default='', help='Path bass_midi_keymap.json (opzionale)')
  ap.add_argument('--duration_s', type=float, default=8.0)
  ap.add_argument('--example_secs', type=float, default=4.0)
  ap.add_argument('--hop_secs',     type=float, default=2.0)
  ap.add_argument('--frame_rate',   type=int,   default=50)
  ap.add_argument('--sample_rate',  type=int,   default=44100)
  ap.add_argument('--onset_blur',   type=int,   default=0)
  ap.add_argument('--override_bpm', type=float, default=None)
  args = ap.parse_args()

  keymap = None
  if args.keymap and os.path.exists(args.keymap):
    keymap = BassMidiKeymap(args.keymap)
    print(f'[keymap] loaded: {args.keymap}')
  else:
    print('[keymap] not provided — KS not separated from notes.')

  print(f'Parsing {args.midi} ...')
  notes, ks_events = parse_midi(args.midi,
                                override_bpm=args.override_bpm,
                                keymap=keymap)
  print(f'Note regolari (pitch 21-64): {len(notes)}')
  print(f'Key switch events:           {len(ks_events)}')
  if notes:
    print(f'Prima nota: t_on={notes[0].t_on:.3f}s '
          f'pitch={notes[0].pitch} vel={notes[0].velocity} '
          f'token={notes[0].token_id}')
    print(f'Ultima:     t_on={notes[-1].t_on:.3f}s '
          f't_off={notes[-1].t_off:.3f}s')
  if ks_events:
    print(f'Primo KS:   t_on={ks_events[0].t_on:.3f}s '
          f'token={ks_events[0].token_id} '
          f'trigger={ks_events[0].trigger}')

  wins = windowed_features_for_file(
      notes, file_duration_s=args.duration_s,
      example_secs=args.example_secs, hop_secs=args.hop_secs,
      frame_rate=args.frame_rate, sample_rate=args.sample_rate,
      onset_blur=args.onset_blur, ks_events=ks_events)
  T = n_frames_for_window(args.example_secs, args.frame_rate, centered=True)
  print(f'\nGenerate {len(wins)} finestre, T_frames={T} ognuna.')
  for i, w in enumerate(wins):
    print(f'\nFinestra {i} (t0={i * args.hop_secs:.2f}s):')
    _smoke_test_summary(w, T)
