import os
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import mido
import pretty_midi
import pysampled

import datanavigator

# Module-level constants
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MIDI_TO_NOTE = {
    pitch_number: f"{NOTE_NAMES[pitch_number % 12]}{(pitch_number // 12) - 1}"
    for pitch_number in range(21, 109)
}
NOTE_TO_MIDI = {note: midi for midi, note in MIDI_TO_NOTE.items()}


def ticks_to_seconds(ticks, tempo, ticks_per_beat):
    """Convert MIDI ticks to seconds based on tempo."""
    seconds_per_beat = tempo / 1_000_000  # Convert µs to seconds
    return (ticks / ticks_per_beat) * seconds_per_beat


def format_time_range(start_dt, duration_seconds, fmt="datetime"):
    """Return (start, end, duration) in the requested format.

    Args:
        start_dt: datetime object for the start time.
        duration_seconds: float, duration in seconds.
        fmt: "datetime" returns datetime objects; "unix" returns float timestamps.

    Returns:
        (start, end, duration) tuple.
    """
    end_dt = start_dt + timedelta(seconds=duration_seconds)
    if fmt == "datetime":
        return start_dt, end_dt, duration_seconds
    elif fmt == "unix":
        return start_dt.timestamp(), end_dt.timestamp(), duration_seconds
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'datetime' or 'unix'.")

class Event(datanavigator.Event):
    def __getitem__(self, key):
        if isinstance(key, int):
            key = MIDI_TO_NOTE[key]
        return self._data[key]
    
class PianoRoll(pysampled.Data):
    pass

class Log(mido.MidiFile):
    def __init__(self, filename=None, file=None, type=1, ticks_per_beat=..., charset='latin1', debug=False, clip=False, tracks=None):
        super().__init__(filename, file, type, ticks_per_beat, charset, debug, clip, tracks)
        self._pretty_midi_object = pretty_midi.PrettyMIDI(filename)
    
    @property
    def time_resolution(self) -> float:
        """Number of seconds per tick."""
        return self.ticks_to_seconds(ticks=1)
    
    @property
    def sr(self):
        """Return the sampling rate."""
        return 1 / self.time_resolution
    
    @property
    def pm(self):
        """Return the PrettyMIDI object."""
        return self._pretty_midi_object
    
    @property
    def notes(self):
        """Return the list of notes from PrettyMidi."""
        return self.pm.instruments[0].notes
    
    @property
    def duration(self):
        """Return the duration of the MIDI file."""
        tr = self.time_resolution
        return sum([round(x.time/tr) for x in self])*tr
    
    def extract_tempo_changes(self, verbose=False):
        ticks_per_beat = self.ticks_per_beat
        default_tempo = 500_000  # Default MIDI tempo (120 BPM)

        absolute_time = 0  # Accumulate time in ticks
        tempo_changes = {0: default_tempo}  # Dictionary of (time_in_seconds, tempo) changes
        current_tempo = default_tempo  # Start with default tempo

        for msg in self:
            absolute_time += msg.time  # Accumulate time in ticks

            if msg.type == 'set_tempo':
                current_tempo = msg.tempo  # Update tempo
                time_in_seconds = sum(
                    ticks_to_seconds(tc_time, tc_tempo, ticks_per_beat) for tc_time, tc_tempo in tempo_changes.items()
                )
                tempo_changes[absolute_time] = current_tempo
                if verbose:
                    print(f'Tempo Change at {absolute_time} ticks (~{time_in_seconds:.3f} sec): {current_tempo} µs per beat')

        return tempo_changes
    
    def has_tempo_changes(self):
        return len(self.extract_tempo_changes()) > 1
    
    def ticks_to_seconds(self, ticks=1):
        if self.has_tempo_changes():
            raise ValueError('Tempo changes are not supported')
        tempo = set(self.extract_tempo_changes().values())
        tempo = tempo.pop()
        return ticks_to_seconds(ticks, tempo, self.ticks_per_beat)
    
    def get_notes_played(self):
        """Return the played notes."""
        notes = []
        for msg in self:
            if msg.type == 'note_on':
                notes.append(msg.note)
        return [MIDI_TO_NOTE[note] for note in set(notes)]

    def to_event(self):
        """Convert the notes to an Event object in datnavigator."""
        assert len(self.pm.instruments) == 1, 'Only one instrument is supported'
        notes = self.notes
        event_data = {note: [] for note in NOTE_TO_MIDI}
        for note in notes:
            note_name = MIDI_TO_NOTE[note.pitch]
            event_data[note_name].append([note.start, note.end])
            
        return Event.from_data(event_data, pick_action="append")

    def to_signals(self):
        """Convert the notes to signals."""
        signal_names = [MIDI_TO_NOTE.get(x, f"m{x}") for x in range(128)]
        signal_coords = ["velocity"]
        return PianoRoll(self.pm.get_piano_roll(fs=self.sr).T, sr=self.sr, signal_names=signal_names, signal_coords=signal_coords)
    
    def get_note_limits(self):
        """Return the start time of the first note and the end time of the last note in this MIDI file."""
        note_start_times = [note.start for note in self.notes]
        note_end_times = [note.end for note in self.notes]
        return np.min(note_start_times), np.max(note_end_times)

    def get_recording_time_range(self, fmt="unix", utc_offset=None):
        """Return (start_time, end_time, duration) for this MIDI recording.

        Start time is parsed from the track_name meta message (first 15 chars
        as YYYYMMDD_HHMMSS). Falls back to file modification time if unavailable.

        Args:
            fmt: "datetime" returns datetime objects; "unix" returns float timestamps.
            utc_offset: Numeric UTC offset in hours (e.g. -5 for CDT, -4 for EDT).
                Required to get correct unix timestamps, since MIDI timestamps
                are naive (no timezone info).

        Returns:
            (start, end, duration) tuple.
        """
        end_dt = None

        # Try to extract from track_name meta message
        # Note: the track_name timestamp is the *end* time of the recording
        for track in self.tracks:
            for msg in track:
                if msg.type == 'track_name' and msg.name:
                    try:
                        end_dt = datetime.strptime(msg.name[:15], "%Y%m%d_%H%M%S")
                    except (ValueError, IndexError):
                        pass
                    break
            if end_dt is not None:
                break

        # Fallback: file modification time
        if end_dt is None and self.filename is not None:
            end_dt = datetime.fromtimestamp(os.stat(self.filename).st_mtime)

        if end_dt is None:
            raise ValueError("Cannot determine recording time")

        # Attach timezone if utc_offset provided
        if utc_offset is not None:
            tz = timezone(timedelta(hours=utc_offset))
            end_dt = end_dt.replace(tzinfo=tz)

        # Derive start time by subtracting duration from end time
        start_dt = end_dt - timedelta(seconds=self.duration)

        return format_time_range(start_dt, self.duration, fmt)

    def get_polytouch_times(self):
        """Extract polytouch (polyphonic aftertouch) events with timing.

        Returns:
            list of tuples: Each tuple contains (time_in_seconds, note, value, channel)
                - time_in_seconds: Absolute time of the polytouch event
                - note: MIDI note number (0-127)
                - value: Pressure value (0-127)
                - channel: MIDI channel (0-15)
        """
        polytouch_events = []

        for track_num, track in enumerate(self.tracks):
            absolute_time_ticks = 0

            for msg in track:
                absolute_time_ticks += msg.time

                if msg.type == 'polytouch':
                    time_in_seconds = self.ticks_to_seconds(absolute_time_ticks)
                    polytouch_events.append((
                        time_in_seconds,
                        msg.note,
                        msg.value,
                        msg.channel
                    ))

        # Sort by time
        polytouch_events.sort(key=lambda x: x[0])
        return polytouch_events

            # get event with the type of the event
    def get_events(self, event_type, **filters):
        """Get events by type, optionally filtered by message attributes.

        Args:
            event_type: MIDI message type to include (e.g., 'note_on', 'polytouch').
                You may also pass an iterable of types to match any of them.
            **filters: Attribute filters applied to each message. For each key/value:
                - If the message lacks the attribute, the message is skipped.
                - If the value is a callable, it should accept the attribute value and
                  return True to keep the message.
                - If the value is a list/tuple/set/range, the attribute must be a member.
                - Otherwise, the attribute must equal the value.

        Examples:
            get_events('polytouch', note=65)
            get_events('control_change', control=64, value=127)
            get_events(['note_on', 'note_off'], note={60, 62, 64})
        """
        
        # Normalize event types to a set for quick membership checks
        if isinstance(event_type, (list, tuple, set, frozenset)):
            event_types = set(event_type)
        else:
            event_types = {event_type}
        
        def _matches_filters(msg):
            for attr_name, expected in filters.items():
                if not hasattr(msg, attr_name):
                    return False
                actual = getattr(msg, attr_name)
                if callable(expected):
                    if not bool(expected(actual)):
                        return False
                elif isinstance(expected, (list, tuple, set, frozenset, range)):
                    if actual not in expected:
                        return False
                else:
                    if actual != expected:
                        return False
            return True
        
        events = []
        for track_num, track in enumerate(self.tracks):
            absolute_time_ticks = 0

            for msg in track:
                absolute_time_ticks += msg.time

                if msg.type in event_types and _matches_filters(msg):
                    time_in_seconds = self.ticks_to_seconds(absolute_time_ticks)
                    events.append(
                        {
                            "time_in_seconds": time_in_seconds,
                            'msg': msg
                        }
                    )

        # Sort by time
        events.sort(key=lambda x: x["time_in_seconds"])
        return events


    def get_aftertouch_times(self):
        """Extract aftertouch (channel aftertouch) events with timing.

        Returns:
            list of tuples: Each tuple contains (time_in_seconds, value, channel)
                - time_in_seconds: Absolute time of the aftertouch event
                - value: Pressure value (0-127)
                - channel: MIDI channel (0-15)
        """
        aftertouch_events = []

        for track_num, track in enumerate(self.tracks):
            absolute_time_ticks = 0

            for msg in track:
                absolute_time_ticks += msg.time

                if msg.type == 'aftertouch':
                    time_in_seconds = self.ticks_to_seconds(absolute_time_ticks)
                    aftertouch_events.append((
                        time_in_seconds,
                        msg.value,
                        msg.channel
                    ))

        # Sort by time
        aftertouch_events.sort(key=lambda x: x[0])
        return aftertouch_events
    
    def show_roll(self, velocity_lim="auto"):
        """Show the piano roll."""
        piano_roll = self.to_signals() #self.pm.get_piano_roll(fs=self.sr)

        if velocity_lim == "auto":
            vel = [x.velocity for x in self.notes]
        else:
            assert isinstance(velocity_lim, (list, tuple)) and len(velocity_lim) == 2, 'velocity_lim must be a tuple of two values'
            vel = velocity_lim
        norm = Normalize(vmin=min(vel), vmax=max(vel))

        figure, ax = plt.subplots(1, 1, figsize=(10, 6))
        im = ax.imshow(piano_roll().T, aspect='auto', cmap='Blues', origin='lower', norm=norm, extent=[piano_roll.t_start(), piano_roll.t_end(), -0.5, 127.5], interpolation='nearest')
        ax.set_ylabel('Note')
        ax.set_xlabel('Time (s)')
        
        note_limits = self.get_note_limits()
        ax.set_xlim(note_limits[0]-1, note_limits[1]+1) # 1 second padding

        # show C1 to C8
        c1_to_c8_notes = {note: midi for note, midi in NOTE_TO_MIDI.items() if note.startswith('C') and '1' <= note[-1] <= '8' and len(note) == 2}
        ax.set_yticks(list(c1_to_c8_notes.values()), list(c1_to_c8_notes.keys()))
        
        for x in range(129):
            ax.axhline(x-0.5, color='black', linewidth=0.1)
        notes_played_idx = [NOTE_TO_MIDI[x] for x in self.get_notes_played()]
        ax.set_ylim(np.min(notes_played_idx)-0.5, np.max(notes_played_idx)+0.5) # 1/2 note witdth padding for rendering

        plt.colorbar(im, label='MIDI velocity')
        figure.tight_layout()
        plt.show(block=False)
    
