import argparse
import os
import sys
import music21
import mido
import copy
import statistics
from collections import Counter
import google.genai as genai # Corrected import based on common practice
import config

client = genai.Client(api_key=config.API_KEY)
model = "gemini-2.5-flash"


# --- Gemini Helper Function ---
# This function will now only be called once at the very end
def generate_content(prompt_text):
    """Sends a prompt to Gemini and returns the interpreted text."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text,
        )
        return response.text.strip()
    except Exception as e:
        return f"[[GEMINI API ERROR: {e} - Could not generate interpretation for this section.]]"


# --- Utility Function to get pitch class sequence ---
def get_pitch_class_sequence(stream_obj):
    """
    Extracts the sequence of pitch classes from a music21 stream or score.
    Considers all notes in chords.
    """
    pc_sequence = []
    for element in stream_obj.flatten().notesAndRests:
        if isinstance(element, music21.note.Note):
            pc_sequence.append(element.pitch.pitchClass)
        elif isinstance(element, music21.chord.Chord):
            for p in element.pitches:
                pc_sequence.append(p.pitchClass)
    return pc_sequence


# --- Main Analysis Function (kept for context, no changes needed here for this request) ---
def analyze_midi_form(score_object, midi_file_path_for_report):
    """
    Analyzes a music21 score object for musical elements and atonal tendencies.
    Takes the score object directly to allow analysis on reductions or originals.
    `midi_file_path_for_report` is used just for display purposes.
    Returns a dictionary of all analysis findings. Gemini interpretation will be done globally.
    """
    analysis_results = {}
    print(f"\n--- Analyzing Score from: '{os.path.basename(midi_file_path_for_report)}' ---")

    # --- Basic Score Information ---
    print(f"\n--- Basic Score Information ---")
    num_parts = len(score_object.parts)
    print(f"Number of Parts (Tracks): {num_parts}")
    analysis_results['num_parts'] = num_parts

    total_duration = score_object.duration.quarterLength
    if total_duration is not None:
        print(f"Total Duration: {float(total_duration):.2f} Quarter Lengths")
        analysis_results['total_duration'] = float(total_duration)
    else:
        print("Total Duration: Could not determine (might be a stream without explicit end)")
        analysis_results['total_duration'] = "N/A"

    # Flatten the score for consistent counting (all notes, rests, chords across parts)
    flat_score_for_counts = score_object.flatten()

    num_notes = len(flat_score_for_counts.getElementsByClass('Note'))
    num_rests = len(flat_score_for_counts.getElementsByClass('Rest'))
    num_chords = len(flat_score_for_counts.getElementsByClass('Chord'))

    print(f"Total Notes (standalone): {num_notes}")
    print(f"Total Rests: {num_rests}")
    print(f"Total Chords: {num_chords}")
    analysis_results['num_notes'] = num_notes
    analysis_results['num_rests'] = num_rests
    analysis_results['num_chords'] = num_chords

    # --- Extracting Measures and Key/Time Signatures ---
    print(f"\n--- Measures and Basic Metrical/Harmonic Information ---")
    measures = score_object.parts[0].getElementsByClass(
        'Measure') if score_object.parts else score_object.getElementsByClass('Measure')

    analysis_results['num_measures'] = 0
    analysis_results['initial_key'] = "Not found"
    analysis_results['initial_time'] = "Not found"
    analysis_results['key_changes'] = []
    analysis_results['time_changes'] = []

    if not measures:
        print("Could not find explicit measures. Analyzing flattened stream for measure-like segments.")
        flat_score_measures = score_object.flatten().makeMeasures()
        if flat_score_measures:
            measures = flat_score_measures.getElementsByClass('Measure')

    if measures:
        print(f"Number of Measures detected: {len(measures)}")
        analysis_results['num_measures'] = len(measures)
        if len(measures) > 0:
            first_measure = measures[0]
            initial_key = first_measure.getElementsByClass('KeySignature')
            initial_time = first_measure.getElementsByClass('TimeSignature')

            if initial_key:
                print(f"Initial Key Signature: {initial_key[0].sharps} sharps/flats ({initial_key[0].name})")
                analysis_results['initial_key'] = initial_key[0].name
            else:
                print(
                    "Initial Key Signature: Not explicitly found (C Major/A Minor assumed by default by music21 if not specified)")

            if initial_time:
                print(f"Initial Time Signature: {initial_time[0].numerator}/{initial_time[0].denominator}")
                analysis_results['initial_time'] = f"{initial_time[0].numerator}/{initial_time[0].denominator}"
            else:
                print(
                    "Initial Time Signature: Not explicitly found (4/4 assumed by default by music21 if not specified)")

        key_changes = score_object.flatten().getElementsByClass('KeySignature')
        time_changes = score_object.flatten().getElementsByClass('TimeSignature')

        if len(key_changes) > 1:
            print(f"Key Signature Changes detected at offsets:")
            for kc in key_changes:
                change_info = f"Offset {float(kc.offset):.2f}: {kc.sharps} sharps/flats ({kc.name})"
                print(f"  {change_info}")
                analysis_results['key_changes'].append(change_info)

        if len(time_changes) > 1:
            print(f"Time Signature Changes detected at offsets:")
            for tc in time_changes:
                change_info = f"Offset {float(tc.offset):.2f}: {tc.numerator}/{tc.denominator}"
                print(f"  {change_info}")
                analysis_results['time_changes'].append(change_info)
    else:
        print("No measures or measure-like structures found in the score.")

    # --- Basic Melodic and Harmonic Analysis (for form hints) ---
    print(f"\n--- Melodic and Harmonic Insights ---")

    # Analyze the flattened score for melodic/harmonic patterns, as reduction aggregates this
    # all_notes_and_rests_flat = flat_score_for_counts.notesAndRests # Not used directly for top_phrases_output
    # top_phrases_output = [] # Not directly calculated/used in this version for Gemini input.
    # melodic_gemini_interpretation = "" # Removed, will be part of overall prompt

    chord_stream = score_object.chordify()
    chord_names = {}
    for c in chord_stream.flatten().getElementsByClass('Chord'):
        try:
            root_name = c.root().name
            chord_type = c.quality
            chord_key = f"{root_name} {chord_type}"
            chord_names[chord_key] = chord_names.get(chord_key, 0) + 1
        except Exception:
            pass

    top_chords_output = []
    # harmonic_gemini_interpretation = "" # Removed, will be part of overall prompt

    if chord_names:
        print("\nMost Common Chords:")
        sorted_chords = sorted(chord_names.items(), key=lambda item: item[1], reverse=True)

        for chord, count in sorted_chords[:10]:
            top_chords_output.append(f"{chord}: {count} occurrences")
            print(f"  {top_chords_output[-1]}")

        analysis_results['top_chords'] = top_chords_output
    else:
        print("No explicit chords or easily inferable chords found.")
        analysis_results['top_chords'] = []


    # --- Identifying Section Markers (Repeats, Dynamics, Text) ---
    analysis_results['dynamics'] = []
    analysis_results['tempi'] = []
    analysis_results['texts'] = []

    dynamics = score_object.flatten().getElementsByClass('Dynamic')
    if dynamics:
        print(f"  Dynamic changes detected:")
        for dyn in dynamics:
            dyn_info = f"Offset {float(dyn.offset):.2f}: {dyn.name}"
            print(f"    {dyn_info}")
            analysis_results['dynamics'].append(dyn_info)
    else:
        print("  No explicit dynamic changes found.")

    tempi = score_object.flatten().getElementsByClass('MetronomeMark')
    if len(tempi) > 1:
        print(f"  Tempo changes detected:")
        for tempo in tempi:
            tempo_info = f"Offset {float(tempo.offset):.2f}: {tempo.number} bpm"
            print(f"    {tempo_info}")
            analysis_results['tempi'].append(tempo_info)
    elif len(tempi) == 1:
        tempo_info = f"Initial Tempo: {tempi[0].number} bpm (no changes detected)."
        print(f"  {tempo_info}")
        analysis_results['tempi'].append(tempo_info)
    else:
        print("  No explicit tempo indications found.")

    texts = score_object.flatten().getElementsByClass('TextExpression')
    if texts:
        print(f"  Text expressions (potential form labels) found:")
        for text in texts:
            text_info = f"Offset {float(text.offset):.2f}: '{text.content}'"
            print(f"    {text_info}")
            analysis_results['texts'].append(text_info)
    else:
        print("  No text expressions found.")

    # ===============================================
    # --- Atonality Indicators ---
    print(f"\n--- Atonality Indicators ---")
    # atonal_gemini_assessment = "" # Removed, will be part of overall prompt

    # 1. Global Key Correlation
    global_key_correlation_score = None
    try:
        global_key = score_object.analyze('key')
        global_key_correlation_score = global_key.correlationScore
        print(
            f"Global Key Analysis: {global_key.tonic.name} {global_key.mode} (correlation: {global_key_correlation_score:.2f})")
        if global_key_correlation_score < 0.3:
            print("  -> Very low key correlation score suggests strong atonal tendencies.")
        elif global_key_correlation_score < 0.5:
            print("  -> Low key correlation score suggests a weak or ambiguous tonal center, possibly atonal.")
        else:
            print("  -> Strong key correlation suggests a tonal center is present.")
    except Exception as e:
        print(f"Could not perform global key analysis (might be due to very unusual or short piece): {e}")
        print("  -> The inability to find a strong key might also be an atonal indicator.")
        global_key_correlation_score = -1  # Indicate failure
    analysis_results['global_key_correlation_score'] = global_key_correlation_score

    # 2. Pitch Class Distribution Evenness
    pitch_class_counts = {i: 0 for i in range(12)}
    total_notes_for_pc = 0
    for element in flat_score_for_counts.notesAndRests:
        if element.isNote:
            pc = element.pitch.pitchClass
            pitch_class_counts[pc] += 1
            total_notes_for_pc += 1
        elif element.isChord:
            for n_in_chord in element.notes:
                pc = n_in_chord.pitch.pitchClass
                pitch_class_counts[pc] += 1
                total_notes_for_pc += 1

    std_dev_pc = None
    if total_notes_for_pc > 0:
        print("\nPitch Class Distribution (0=C, 1=C#, ..., 11=B):")
        pc_dist_output = []
        for pc, count in sorted(pitch_class_counts.items()):
            percentage = (count / total_notes_for_pc) * 100
            pc_dist_output.append(
                f"PC {pc:2d} ({music21.pitch.Pitch(midi=pc + 60).name.replace('-', 'b')}): {count:4d} notes ({percentage:.1f}%)")
            print(pc_dist_output[-1])

        non_zero_counts = [c for c in pitch_class_counts.values() if c > 0]
        if len(non_zero_counts) > 1:
            std_dev_pc = statistics.stdev(non_zero_counts)
            print(f"  Standard Deviation of Pitch Class Counts: {std_dev_pc:.2f}")
            if std_dev_pc < 10:
                print(
                    "  -> Low standard deviation suggests a relatively even distribution of pitch classes, common in atonal music.")
            else:
                print(
                    "  -> Higher standard deviation suggests an uneven distribution, indicating tonal hierarchies or emphasis on certain pitches.")
        elif len(non_zero_counts) == 1:
            print("  Only one pitch class found, cannot calculate meaningful standard deviation.")
        else:
            print("  No notes found to calculate pitch class distribution.")
    else:
        print("  No notes found for pitch class distribution analysis.")
    analysis_results['std_dev_pc'] = std_dev_pc
    analysis_results['pc_dist_output'] = pc_dist_output

    # 3. Prevalence of Dissonant Melodic Intervals
    print("\nMelodic Interval Distribution:")

    all_individual_notes_in_time = []
    for element in flat_score_for_counts.notesAndRests:
        if element.isNote:
            all_individual_notes_in_time.append(element)
        elif element.isChord:
            # Sort notes within a chord by pitch to simulate vertical "melodic" connections or density
            # This is a common way to flatten chords into a linear sequence for interval analysis
            for n_in_chord in sorted(element.notes, key=lambda x: x.pitch.midi):
                all_individual_notes_in_time.append(n_in_chord)

    all_individual_notes_in_time.sort(key=lambda x: x.offset)

    melodic_intervals = {}
    last_note = None

    for current_note in all_individual_notes_in_time:
        if current_note.isNote:  # Ensure we are comparing actual notes
            if last_note is not None and current_note.offset >= last_note.offset:  # Handle simultaneous or sequential
                # For simultaneous notes (same offset, e.g., from a chord), consider them as vertical intervals or dense texture
                # For strictly melodic, current_note.offset > last_note.offset
                if current_note.offset > last_note.offset:  # Strictly melodic
                    try:
                        interval_obj = music21.interval.Interval(noteStart=last_note, noteEnd=current_note)
                        interval_name = interval_obj.name
                        melodic_intervals[interval_name] = melodic_intervals.get(interval_name, 0) + 1
                    except Exception as e:
                        pass
                elif current_note.offset == last_note.offset and len(all_individual_notes_in_time) > 1:
                    # If simultaneous and not the first element, consider interval between last and current within a chord context
                    # This adds a layer of "harmonic" interval thinking within the "melodic" flow
                    try:
                        interval_obj = music21.interval.Interval(noteStart=last_note, noteEnd=current_note)
                        interval_name = interval_obj.name + " (simultaneous)"  # Mark as simultaneous
                        melodic_intervals[interval_name] = melodic_intervals.get(interval_name, 0) + 1
                    except Exception as e:
                        pass
            last_note = current_note

    dissonant_percentage = 0
    interval_report = []

    if melodic_intervals:
        sorted_intervals = sorted(melodic_intervals.items(), key=lambda item: item[1], reverse=True)
        print("  Top 10 Melodic/Simultaneous Intervals:")

        for interval_str, count in sorted_intervals[:10]:
            interval_report.append(f"{interval_str}: {count} occurrences")
            print(f"    {interval_report[-1]}")

        dissonant_semitone_values = {1, 2, 6, 10, 11}  # m2, M2, TT, m7, M7 (and their compounds via simpleSemitones)
        total_intervals = sum(melodic_intervals.values())
        dissonant_count = 0

        for interval_str, count in melodic_intervals.items():
            try:
                # Remove "(simultaneous)" for interval parsing
                clean_interval_str = interval_str.replace(" (simultaneous)", "")
                interval_obj_from_str = music21.interval.Interval(clean_interval_str)
                # Check for simple and compound dissonant intervals based on semitones
                if interval_obj_from_str.semitones is not None and \
                        (abs(interval_obj_from_str.semitones % 12) in dissonant_semitone_values or
                         abs(interval_obj_from_str.simpleSemitones) in dissonant_semitone_values):
                    dissonant_count += count
            except Exception:
                pass

        if total_intervals > 0:
            dissonant_percentage = (dissonant_count / total_intervals) * 100
            print(
                f"\n  Percentage of Dissonant Melodic/Simultaneous Intervals (m2, M2, TT, m7, M7): {dissonant_percentage:.1f}%")
            if dissonant_percentage > 45:
                print("  -> High percentage of dissonant intervals is a strong indicator of atonal music.")
            elif dissonant_percentage > 30:
                print(
                    "  -> Significant presence of dissonant intervals, potentially atonal or highly chromatic.")
            else:
                print("  -> Lower percentage of dissonant intervals, common in tonal music.")
        else:
            print("  Not enough intervals found for dissonance analysis.")
    else:
        print(
            "  No melodic/simultaneous intervals found for analysis. (This might happen if all notes are simultaneous or the piece is too short.)")
    analysis_results['dissonant_percentage'] = dissonant_percentage
    analysis_results['interval_report'] = interval_report

    return analysis_results


def find_split_offsets_by_12_tone_completeness(score_object, midi_file_name_for_report):
    """
    Analyzes the music21 score to find offsets where all 12 pitch classes
    have been encountered since the last reset.

    Args:
        score_object (music21.stream.Score or music21.stream.Part): The music21 score object.
        midi_file_name_for_report (str): Name of the MIDI file (for reporting).

    Returns:
        list: A sorted list of offsets (in quarter lengths) where splits should occur.
              These offsets mark the *end of the last event* that completed a 12-tone set.
              The splitting function will then include this event.
    """
    print(f"\n--- Identifying Split Offsets by 12 Pitch Class Completeness on '{midi_file_name_for_report}' ---")
    split_offsets = []
    encountered_pitch_classes = set()

    # Store the element that completed the 12-tone set
    last_completing_element = None

    # We need to process all elements in time order and get their original offsets.
    # Flattening and then sorting by offset ensures this.
    # Using notesAndRests ensures we get both notes and chords properly
    all_elements_sorted = sorted(score_object.flatten().notesAndRests, key=lambda el: el.offset)

    for i, element in enumerate(all_elements_sorted):
        # Only consider notes or chords for pitch class
        if isinstance(element, (music21.note.Note, music21.chord.Chord)):
            if isinstance(element, music21.note.Note):
                pc = element.pitch.pitchClass
                if pc not in encountered_pitch_classes:
                    encountered_pitch_classes.add(pc)
                    if len(encountered_pitch_classes) == 12:
                        last_completing_element = element  # This element completed the set
            elif isinstance(element, music21.chord.Chord):
                for p in element.pitches:
                    pc = p.pitchClass
                    if pc not in encountered_pitch_classes:
                        encountered_pitch_classes.add(pc)
                        if len(encountered_pitch_classes) == 12:
                            last_completing_element = element  # This element completed the set

        if last_completing_element is not None and len(encountered_pitch_classes) == 12:
            # We found all 12 tones. The split should occur *after* this element finishes.
            # So, the offset is the start of the element + its duration.
            split_point_ql = float(last_completing_element.offset + last_completing_element.duration.quarterLength)
            split_offsets.append(split_point_ql)
            print(
                f"  Found all 12 pitch classes at offset: {float(last_completing_element.offset):.2f} quarter lengths (ends at {split_point_ql:.2f}). Setting split point.")
            encountered_pitch_classes = set()  # Reset for the next segment
            last_completing_element = None  # Reset the completing element

    if not split_offsets:
        print("  No segments containing all twelve pitch classes were found in the piece.")
    else:
        # Remove duplicates and sort, though they should be sorted already from the loop
        split_offsets = sorted(list(set(split_offsets)))
        print(f"  Identified {len(split_offsets)} potential split points based on 12-tone completeness.")

    return split_offsets


def split_midi_file_at_offsets(input_midi_file_path, split_offsets_ql, output_directory="split_midi_files_12tone"):
    """
    Splits a MIDI file into smaller files based on a list of quarter length offsets.
    Each split file contains events up to, and including, the 12th tone (the event at the split offset).

    Args:
        input_midi_file_path (str): The path to the input MIDI file.
        split_offsets_ql (list): A list of offsets (in quarter lengths) where splits should occur.
                                 These offsets mark the *end* of the event that completed the 12-tone set.
        output_directory (str): The directory where the split MIDI files will be saved.
    Returns:
        list: A list of paths to the created split MIDI files.
    """
    try:
        mid_original = mido.MidiFile(input_midi_file_path)
    except Exception as e:
        print(f"Error loading MIDI file {input_midi_file_path}: {e}")
        return []

    os.makedirs(output_directory, exist_ok=True)
    print(
        f"\n--- Splitting MIDI using Mido based on identified offsets for '{os.path.basename(input_midi_file_path)}' ---")

    ticks_per_quarter = mid_original.ticks_per_beat

    # Convert quarter lengths to ticks. These are the ABSOLUTE TICKS of the end of the 12th note's duration.
    # We add a tiny epsilon to ensure that the exact tick from music21 is included.
    split_event_inclusive_end_ticks = sorted(
        [int(round(offset_ql * ticks_per_quarter + 1e-6)) for offset_ql in split_offsets_ql])

    print(f"  Original MIDI Ticks per Beat (Quarter Note): {ticks_per_quarter}")
    print(f"  Split offsets (quarter lengths, for 12th note END): {split_offsets_ql}")
    print(f"  Split offsets (in rounded, inclusive end ticks): {split_event_inclusive_end_ticks}")

    # Build a list of all messages with their absolute times and original track index
    all_messages_with_absolute_time = []
    for track_idx, track in enumerate(mid_original.tracks):
        absolute_time_in_track = 0
        for msg in track:
            absolute_time_in_track += msg.time
            all_messages_with_absolute_time.append((absolute_time_in_track, track_idx, msg))

    # Sort all messages by their absolute time. Then by track index for consistent ordering of simultaneous events.
    all_messages_with_absolute_time.sort(key=lambda x: (x[0], x[1]))

    segment_file_count = 0
    created_segment_paths = []
    base_filename_no_ext = os.path.basename(input_midi_file_path).replace('.mid', '')

    # Helper function to save a segment (same as previous, with robust empty check)
    def save_segment(segment_data, start_tick, end_tick_for_report, file_num, base_filename):
        has_meaningful_data = False
        for _, _, msg in segment_data:
            if msg.type in ['note_on', 'note_off', 'control_change', 'pitchwheel', 'program_change', 'set_tempo']:
                has_meaningful_data = True
                break

        if not has_meaningful_data:
            print(f"  DEBUG: Skipping saving empty or meaningless segment {file_num} (no notes/relevant data).")
            return None

        new_mid = mido.MidiFile(type=mid_original.type, ticks_per_beat=mid_original.ticks_per_beat)
        tracks_data = {i: [] for i in range(len(mid_original.tracks))}

        for abs_time, track_idx, msg in segment_data:
            tracks_data[track_idx].append((abs_time, msg))

        for track_idx in sorted(tracks_data.keys()):
            new_track = mido.MidiTrack()
            track_events = tracks_data[track_idx]

            if not track_events:
                # Add an end_of_track message if the track is empty, otherwise mido might complain
                new_track.append(mido.MetaMessage('end_of_track', time=0))
                new_mid.tracks.append(new_track)
                continue

            # Calculate relative times
            last_abs_time = start_tick  # Important: base delta time on segment start or previous event
            for abs_time, msg in track_events:
                delta_time = abs_time - last_abs_time
                # Ensure delta_time is non-negative.
                if delta_time < 0:
                    delta_time = 0  # This case should ideally not happen if messages are sorted and start_tick is correct
                new_msg = msg.copy(time=delta_time)
                new_track.append(new_msg)
                last_abs_time = abs_time

            # Ensure all tracks end with end_of_track meta message
            if not any(msg.type == 'end_of_track' for msg in new_track):
                new_track.append(mido.MetaMessage('end_of_track', time=0))

            new_mid.tracks.append(new_track)

        output_filename = os.path.join(output_directory,
                                       f"{base_filename.replace('.mid', '')}_segment_{file_num:03d}.mid")
        try:
            new_mid.save(output_filename)
            print(f"  Saved segment {file_num}: {output_filename} (from tick {start_tick} to {end_tick_for_report})")
            return output_filename
        except Exception as e:
            print(f"  Error saving {output_filename}: {e}")
            return None

    current_midi_message_index = 0
    segment_start_tick = 0

    # If no split points are provided, treat the entire file as one segment.
    if not split_event_inclusive_end_ticks:
        # Define a single 'split point' at the very end of the file.
        if all_messages_with_absolute_time:
            # The last event's absolute time + its duration (assume min 1 tick duration if not a note_off)
            # A more robust way might be to parse with music21 to get exact end of last element.
            # For mido, last message's absolute time is the most practical 'end'.
            # It's better to capture all relevant messages, including metadata at the start of the file.
            # So, the end tick should be the absolute time of the last message in the file.
            split_event_inclusive_end_ticks = [all_messages_with_absolute_time[-1][0]]
            print("  No 12-tone completion points. Treating entire file as one segment.")
        else:
            print("  No messages in file and no split points. No segments to create.")
            return []

    # Iterate through each defined split point (which is the inclusive end tick for a segment)
    for i, segment_end_tick_boundary in enumerate(split_event_inclusive_end_ticks):

        current_segment_events = []
        last_event_tick_in_segment = segment_start_tick  # For reporting

        print(
            f"\n  DEBUG: --- Preparing Segment {segment_file_count} (targeting end tick: {segment_end_tick_boundary}) ---")
        print(f"  DEBUG: Current message index pointer: {current_midi_message_index}")
        print(f"  DEBUG: Current segment starts at tick: {segment_start_tick}")

        # Collect messages for the current segment
        # We collect messages whose absolute time (start tick) is strictly less than the segment_end_tick_boundary,
        # OR messages whose absolute time is *equal* to the boundary if it's the very last event
        # that completes the 12-tone set.
        # The split_offsets_ql now represents the END of the 12th tone event.
        # So we want to include all messages whose ABSOLUTE START TIME is less than or equal to this END.
        while current_midi_message_index < len(all_messages_with_absolute_time):
            abs_time, track_idx, msg = all_messages_with_absolute_time[current_midi_message_index]

            # The logic here is that segment_end_tick_boundary represents the *last tick*
            # that should be included in the current segment.
            # So, if an event starts at `abs_time` and `abs_time <= segment_end_tick_boundary`, it's included.
            if abs_time <= segment_end_tick_boundary:
                current_segment_events.append((abs_time, track_idx, msg))
                last_event_tick_in_segment = abs_time
                current_midi_message_index += 1
            else:
                # This message is at or beyond the current segment's boundary.
                # It belongs to the next segment (or the remainder).
                print(
                    f"  DEBUG: Event at tick {abs_time} is beyond current segment boundary {segment_end_tick_boundary}. Breaking collection for current segment.")
                break  # Exit this inner collection loop

        # After collecting, save the segment
        saved_path = save_segment(current_segment_events, segment_start_tick,
                                  last_event_tick_in_segment, segment_file_count, base_filename_no_ext)
        if saved_path:
            created_segment_paths.append(saved_path)

        # Prepare for the next segment
        segment_file_count += 1
        # The next segment starts from the tick of the first event NOT included in the current segment.
        # This means `all_messages_with_absolute_time[current_midi_message_index]`[0]
        if current_midi_message_index < len(all_messages_with_absolute_time):
            segment_start_tick = all_messages_with_absolute_time[current_midi_message_index][0]
        else:
            # If all messages have been processed, set start_tick to last event tick + 1 (or just break)
            segment_start_tick = last_event_tick_in_segment + 1

        # If we reached the end of all messages, no more segments to process
        if current_midi_message_index >= len(all_messages_with_absolute_time):
            print("  DEBUG: All MIDI messages processed.")
            break

    # Final segment catch-all: If there are remaining messages after the last explicit split point,
    # collect them into a final segment. This handles the end of the file correctly.
    if current_midi_message_index < len(all_messages_with_absolute_time):
        print(f"\n  DEBUG: --- Final Segment Catch-All ---")
        print(f"  DEBUG: Remaining messages from index {current_midi_message_index} to end of file.")

        final_segment_events = all_messages_with_absolute_time[current_midi_message_index:]
        final_segment_last_tick = all_messages_with_absolute_time[-1][0]

        saved_path = save_segment(final_segment_events, segment_start_tick,
                                  final_segment_last_tick, segment_file_count, base_filename_no_ext)
        if saved_path:
            created_segment_paths.append(saved_path)
        segment_file_count += 1  # Increment for completeness

    print(f"\nMIDI splitting process completed. Total {len(created_segment_paths)} actual segments saved.")
    print(f"Check the '{os.path.join(os.getcwd(), output_directory)}' directory for the split files.")
    return created_segment_paths


def analyze_twelve_tone_row_in_segment(segment_file_path):
    """
    Analyzes a given MIDI segment (loaded as a music21 score) for the presence of
    twelve-tone rows according to Schoenberg's rules, and explores other post-tonal techniques.
    Returns a dictionary of analysis findings for this segment. Gemini interpretation will be done globally.
    """
    segment_analysis_results = {'segment_path': segment_file_path}
    print(
        f"\n--- Analyzing '{os.path.basename(segment_file_path)}' for Twelve-Tone Rows and Other Post-Tonal Techniques ---")
    try:
        segment_score = music21.converter.parse(segment_file_path)
    except Exception as e:
        print(f"  Error parsing segment {segment_file_path}: {e}")
        segment_analysis_results['error'] = f"Error parsing segment: {e}"
        return segment_analysis_results

    # Get the raw pitch class sequence from the segment
    pc_sequence_raw = get_pitch_class_sequence(segment_score)
    segment_analysis_results['pc_sequence_raw'] = pc_sequence_raw

    if len(pc_sequence_raw) < 12:
        print("  Segment too short to contain a full 12-tone row.")
        segment_analysis_results['segment_too_short'] = True
        return segment_analysis_results

    # Find candidate prime rows: iterate through all possible 12-note contiguous subsequences
    # and check if they contain exactly 12 unique pitch classes.
    candidate_rows = []

    for i in range(len(pc_sequence_raw) - 11):
        sub_sequence = pc_sequence_raw[i: i + 12]

        if len(set(sub_sequence)) == 12:
            candidate_rows.append(sub_sequence)
            print(f"  Identified a potential 12-tone row (all 12 unique PCs) at index {i}: {sub_sequence}")
        else:
            pass # No need to print rejection reasons extensively, the focus is on finding candidates.

    if not candidate_rows:
        print("  No contiguous 12-tone rows (all 12 unique pitch classes) found in this segment.")
        print("  (A strict 12-tone row must contain each of the 12 pitch classes exactly once.)")
        segment_analysis_results['no_strict_contiguous_rows'] = True
    else:  # If candidate_rows were found
        # Let's take the first complete candidate row as our "prime row" for this segment's analysis
        potential_prime_row_pcs = candidate_rows[0]
        segment_analysis_results['potential_prime_row_pcs'] = potential_prime_row_pcs

        # Create a music21.serial.ToneRow object for the potential prime
        try:
            prime_row_obj = music21.serial.ToneRow(potential_prime_row_pcs)
            print(f"  Potential Prime Row (P0) for this segment: {prime_row_obj.pitchClasses}")
        except Exception as e:
            print(f"  Could not create music21.serial.ToneRow from candidate: {potential_prime_row_pcs}. Error: {e}")
            segment_analysis_results['error'] = f"Could not create ToneRow: {e}"
            return segment_analysis_results

        # Generate all 48 forms of this potential prime row
        all_row_forms_pcs = {}

        for t in range(12):
            p_t = prime_row_obj.originalCenteredTransformation('T', t)
            all_row_forms_pcs[('P', t)] = p_t.pitchClasses
            i_t = prime_row_obj.originalCenteredTransformation('I', t)
            all_row_forms_pcs[('I', t)] = i_t.pitchClasses
            r_t = prime_row_obj.originalCenteredTransformation('R', t)
            all_row_forms_pcs[('R', t)] = r_t.pitchClasses
            ri_t = prime_row_obj.originalCenteredTransformation('RI', t)
            all_row_forms_pcs[('RI', t)] = ri_t.pitchClasses

        print(f"  Generated {len(all_row_forms_pcs)} possible row forms from the candidate prime row.")

        # Now, try to find occurrences of these 48 forms within the segment's raw pitch class sequence
        found_rows_in_segment = Counter()

        i = 0
        while i <= len(pc_sequence_raw) - 12:
            sub_sequence = pc_sequence_raw[i: i + 12]

            if len(set(sub_sequence)) == 12:  # Only check subsequences that are 12-tone aggregates
                matched_form = None
                for (form_type, trans_idx), row_pcs in all_row_forms_pcs.items():
                    if sub_sequence == row_pcs:
                        matched_form = (form_type, trans_idx)
                        break

                if matched_form:
                    found_rows_in_segment[matched_form] += 1
                    print(f"    Found row form {matched_form} at offset index {i}: {sub_sequence}")
                    i += 12  # Move window to just after the found row
                else:
                    i += 1  # No strict match, slide window by one
            else:
                i += 1  # Not a unique 12-tone sequence, slide window by one

        if found_rows_in_segment:
            print(f"\n  Summary of Schoenberg-like 12-tone row forms found in '{os.path.basename(segment_file_path)}':")
            row_findings_report = []
            for (form_type, trans_idx), count in found_rows_in_segment.most_common():
                row_findings_report.append(
                    f"Form {form_type}{trans_idx} ({music21.pitch.Pitch(midi=trans_idx + 60).name.replace('-', 'b')}): {count} occurrence(s)")
                print(f"    {row_findings_report[-1]}")
            segment_analysis_results['found_rows_in_segment_summary'] = row_findings_report

        else:  # No strict Schoenbergian rows found, even if 12-tone aggregate was identified
            print(
                f"  No Schoenberg-like 12-tone row forms found in '{os.path.basename(segment_file_path)}' based on the derived prime row.")
            print("  This segment may contain 12 pitch classes, but not in a strict, ordered row application,")
            print("  or the inferred prime row may not be representative of the segment's serial organization.")
            segment_analysis_results['no_strict_row_forms_from_prime'] = True

    return segment_analysis_results


def generate_piano_reduction(original_midi_file_path, output_directory="piano_reductions"):
    """
    Generates a piano reduction MIDI file by flattening the original score
    and preserving all contextual elements.

    Args:
        original_midi_file_path (str): Path to the input MIDI file.
        output_directory (str): Directory to save the piano reduction.
    Returns:
        tuple: (path to the generated piano reduction MIDI file, music21.Score object of the reduction)
               or (None, None) if reduction not created/necessary.
    """
    print(f"\n--- Attempting to generate piano reduction for '{os.path.basename(original_midi_file_path)}' ---")
    try:
        original_score = music21.converter.parse(original_midi_file_path)

        if len(original_score.parts) <= 1:
            print("  MIDI file has 1 or zero parts. No multi-instrument reduction necessary.")
            # Still return the original score parsed for consistency in analysis pipeline
            return original_midi_file_path, original_score

        # Create a new Score for the reduction
        piano_reduction_score = music21.stream.Score()
        piano_reduction_score.insert(0, music21.metadata.Metadata())
        piano_reduction_score.metadata.title = f"Piano Reduction of {original_score.metadata.title or os.path.basename(original_midi_file_path)}"
        piano_reduction_score.metadata.composer = original_score.metadata.composer

        # Create a single Part for the piano
        piano_part = music21.stream.Part()
        piano_part.id = 'Piano Reduction'
        piano_part.insert(0, music21.instrument.Piano())  # Assign Piano instrument

        # Flatten the original score to get all musical events in one stream
        # This will combine notes into chords if they are simultaneous.
        # It also attempts to preserve all elements like time/key signatures,
        # dynamics, tempo marks, articulations are transferred.
        for el in original_score.flatten():
            piano_part.insert(el.offset, el)

        piano_reduction_score.append(piano_part)

        output_filename = os.path.join(output_directory,
                                       f"{os.path.basename(original_midi_file_path).replace('.mid', '')}_piano_reduction.mid")
        os.makedirs(output_directory, exist_ok=True)

        piano_reduction_score.write('midi', fp=output_filename)
        print(f"  Piano reduction saved to: {output_filename}")
        return output_filename, piano_reduction_score

    except Exception as e:
        print(f"  Error generating piano reduction: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a MIDI file, perform musical analysis, split it into smaller files "
                    "every time all 12 pitch classes have been used, analyze each segment "
                    "for Schoenberg's dodecaphonic rules, and optionally generate a piano reduction.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "midi_file",
        help="Path to the MIDI file to analyze and split.\n"
             "Example: python your_script_name.py my_tune.mid"
    )
    args = parser.parse_args()

    if not os.path.exists(args.midi_file):
        print(f"Error: MIDI file not found at '{args.midi_file}'")
        sys.exit(1)
    author = input("Enter composition title and author's name> ")
    # --- Initial parsing of the input MIDI file ---
    original_score_object = None
    try:
        original_score_object = music21.converter.parse(args.midi_file)
        print(f"\nOriginal MIDI file '{args.midi_file}' successfully loaded for initial inspection.")
        # Removed "autor" input as it will be part of the final prompt now
        # autor = input("Enter name of composition and its author: ")
    except music21.converter.ConverterException as e:
        print(f"\nError: music21 could not parse the original MIDI file '{args.midi_file}'. Details: {e}")
        print("This might indicate a corrupted MIDI file or an incompatibility with music21.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred while loading the original MIDI: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    source_midi_for_analysis_path = args.midi_file
    source_score_for_analysis = original_score_object
    analysis_source_description = "original MIDI file"

    # Check if a piano reduction is needed
    if len(original_score_object.parts) > 1:
        print(
            f"\nDetected multiple parts ({len(original_score_object.parts)}) in the original MIDI. Generating piano reduction for analysis.")
        reduction_path, reduction_score = generate_piano_reduction(args.midi_file)
        if reduction_path and reduction_score:
            source_midi_for_analysis_path = reduction_path
            source_score_for_analysis = reduction_score
            analysis_source_description = "piano reduction"
        else:
            print("  Piano reduction could not be generated. Proceeding with analysis on the original MIDI file.")
    else:
        print("\nOriginal MIDI file has 1 or zero parts. Analysis will proceed on the original MIDI file.")

    # --- Perform general analysis on the chosen source ---
    general_analysis_results = analyze_midi_form(source_score_for_analysis, source_midi_for_analysis_path)

    # --- Get the offsets where splits should occur based on 12-tone completeness from the chosen source ---
    split_points = find_split_offsets_by_12_tone_completeness(source_score_for_analysis,
                                                              os.path.basename(source_midi_for_analysis_path))

    # --- Then, use mido to split the chosen MIDI file at these points ---
    created_segments = split_midi_file_at_offsets(source_midi_for_analysis_path, split_points)

    # --- Now, analyze each created segment for 12-tone rows ---
    all_segment_analysis_results = []
    if created_segments:
        print(f"\n--- Starting 12-Tone Row Analysis for Each Segment from the {analysis_source_description} ---")
        for segment_path in created_segments:
            segment_analysis_data = analyze_twelve_tone_row_in_segment(segment_path)
            all_segment_analysis_results.append(segment_analysis_data)
    else:
        print(f"\nNo segments were created from the {analysis_source_description}, skipping 12-tone row analysis.")

    print("\nOverall process completed.")

    # --- Generate Final Gemini Summary with ALL data ---
    final_summary_prompt_parts = [
        f"Conduct a comprehensive musicological analysis of the piece represented by the MIDI file '{author}'.\n\n",
        f"Integrate all the following analysis findings into a cohesive, academic report. Address the piece's form, tonality, structure, compositional techniques, and historical context as inferred from the data. Specifically, discuss any evidence of functional tonality, modality, tonal centers, and how these might coexist with or be challenged by atonal or serial elements.\n\n"
    ]

    # Add general analysis results
    final_summary_prompt_parts.append("## General Musical Analysis Highlights\n")
    final_summary_prompt_parts.append(f"**Source for Analysis**: {analysis_source_description.capitalize()}\n")
    final_summary_prompt_parts.append(f"**Basic Score Information**:\n")
    final_summary_prompt_parts.append(f"- Number of Parts: {general_analysis_results.get('num_parts', 'N/A')}\n")
    final_summary_prompt_parts.append(
        f"- Total Duration: {general_analysis_results.get('total_duration', 'N/A')} Quarter Lengths\n")
    final_summary_prompt_parts.append(
        f"- Total Notes (standalone): {general_analysis_results.get('num_notes', 'N/A')}\n")
    final_summary_prompt_parts.append(f"- Total Chords: {general_analysis_results.get('num_chords', 'N/A')}\n")
    final_summary_prompt_parts.append(f"- Number of Measures: {general_analysis_results.get('num_measures', 'N/A')}\n")
    final_summary_prompt_parts.append(
        f"- Initial Key Signature: {general_analysis_results.get('initial_key', 'Not found')}\n")
    final_summary_prompt_parts.append(
        f"- Initial Time Signature: {general_analysis_results.get('initial_time', 'Not found')}\n")
    if general_analysis_results.get('key_changes'):
        final_summary_prompt_parts.append(f"- Key Changes: {'; '.join(general_analysis_results['key_changes'])}\n")
    if general_analysis_results.get('time_changes'):
        final_summary_prompt_parts.append(f"- Time Changes: {'; '.join(general_analysis_results['time_changes'])}\n")
    if general_analysis_results.get('dynamics'):
        final_summary_prompt_parts.append(f"- Dynamic Changes: {'; '.join(general_analysis_results['dynamics'])}\n")
    if general_analysis_results.get('tempi'):
        final_summary_prompt_parts.append(f"- Tempo Indications: {'; '.join(general_analysis_results['tempi'])}\n")
    if general_analysis_results.get('texts'):
        final_summary_prompt_parts.append(
            f"- Text Expressions (potential form labels): {'; '.join(general_analysis_results['texts'])}\n")

    final_summary_prompt_parts.append(f"\n**Melodic & Harmonic Insights**:\n")
    final_summary_prompt_parts.append(
        f"- Most Common Chords: {', '.join(general_analysis_results.get('top_chords', ['No explicit chords']))}\n")

    final_summary_prompt_parts.append(f"\n**Atonality Indicators**:\n")
    final_summary_prompt_parts.append(
        f"- Global Key Correlation Score: {general_analysis_results.get('global_key_correlation_score', 'N/A'):.2f}\n")
    final_summary_prompt_parts.append(
        f"- Pitch Class Distribution Std Dev: {general_analysis_results.get('std_dev_pc', 'N/A'):.2f}\n")
    final_summary_prompt_parts.append(f"- Pitch Class Distribution:\n  " + "\n  ".join(general_analysis_results.get('pc_dist_output', ['N/A'])) + "\n")
    final_summary_prompt_parts.append(
        f"- Percentage of Dissonant Intervals: {general_analysis_results.get('dissonant_percentage', 'N/A'):.1f}%\n")
    final_summary_prompt_parts.append(f"- Top 10 Melodic/Simultaneous Intervals:\n  " + "\n  ".join(general_analysis_results.get('interval_report', ['N/A'])) + "\n")

    # Add Twelve-Tone Analysis Summary
    final_summary_prompt_parts.append("\n## Twelve-Tone Analysis by Segment\n")
    if all_segment_analysis_results:
        final_summary_prompt_parts.append(
            f"The piece was split into {len(all_segment_analysis_results)} segments based on the completion of 12-pitch-class aggregates. Here are the findings for each segment:\n")
        for i, segment_data in enumerate(all_segment_analysis_results):
            final_summary_prompt_parts.append(
                f"\n### Segment {i + 1} ({os.path.basename(segment_data.get('segment_path', 'N/A'))})\n")
            if segment_data.get('error'):
                final_summary_prompt_parts.append(f"- Error during analysis: {segment_data['error']}\n")
                continue

            final_summary_prompt_parts.append(
                f"- Raw Pitch Class Sequence (first 50): {str(segment_data.get('pc_sequence_raw', [])[:50])}...\n")

            if segment_data.get('segment_too_short'):
                final_summary_prompt_parts.append(f"- Segment too short to contain a full 12-tone row.\n")
            elif segment_data.get('no_strict_contiguous_rows'):
                final_summary_prompt_parts.append(
                    f"- No contiguous 12-tone rows (all 12 unique pitch classes) found in this segment (implies free atonality, pitch-class set theory, or other non-strict serial techniques).\n")
            else:
                final_summary_prompt_parts.append(
                    f"- Inferred Prime Row (P0): {segment_data.get('potential_prime_row_pcs', 'N/A')}\n")

                if segment_data.get('found_rows_in_segment_summary'):
                    final_summary_prompt_parts.append(
                        f"- Identified Row Forms: {'; '.join(segment_data['found_rows_in_segment_summary'])}\n")
                elif segment_data.get('no_strict_row_forms_from_prime'):
                    final_summary_prompt_parts.append(
                        f"- No Schoenberg-like 12-tone row forms (P, I, R, RI) identified from the inferred prime row, despite containing all 12 pitch classes. This suggests potential free application within aggregates or different serial organization.\n")

    else:
        final_summary_prompt_parts.append("No 12-tone segments were identified or analyzed.\n")

    final_summary_prompt_parts.append("\n## Overall Musicological Conclusion\n")
    final_summary_prompt_parts.append(
        "Based on all the provided data from the 'General Musical Analysis Highlights' and 'Twelve-Tone Analysis by Segment' sections, provide a comprehensive, overarching musicological conclusion about the piece. Discuss its genre, style, and any notable compositional techniques. Summarize its tonal/atonal characteristics and the extent of any serial or twelve-tone applications. Explicitly address any evidence of functional tonality, modality, or tonal centers, and how these might coexist with or be challenged by other compositional approaches. Frame your response as a cohesive academic report. Consider the context of a potential composer (if known, otherwise a generic composer of such music).")

    final_summary_text = ""
    try:
        final_summary_text = generate_content("\n".join(final_summary_prompt_parts))
        print("\n" + "=" * 80)
        print("FINAL GEMINI MUSICOLOGICAL SUMMARY REPORT")
        print("=" * 80)
        print(final_summary_text)
        print("=" * 80 + "\n")

        # Save to file
        output_report_dir = "analysis_reports"
        os.makedirs(output_report_dir, exist_ok=True)
        report_filename = os.path.join(output_report_dir,
                                       f"{os.path.basename(args.midi_file).replace('.mid', '')}_analysis_report.txt")
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_summary_text)
        print(f"Full analysis report saved to: {os.path.abspath(report_filename)}")

    except Exception as e:
        print(f"\nError generating final Gemini summary: {e}")
        final_summary_text = f"Failed to generate final summary due to an error: {e}"