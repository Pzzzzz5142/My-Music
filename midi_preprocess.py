import argparse
from glob import glob
from threading import Thread
import pretty_midi, os
from multiprocessing import Pool
import json
from tqdm import tqdm

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100

START_IDX = {
    "note_on": 0,
    "note_off": RANGE_NOTE_ON,
    "time_shift": RANGE_NOTE_ON + RANGE_NOTE_OFF,
    "velocity": RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,
}


class SustainAdapter:
    def __init__(self, time, type):
        self.start = time
        self.type = type


class SustainDownManager:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.managed_notes = []
        self._note_dict = {}  # key: pitch, value: note.start

    def add_managed_note(self, note: pretty_midi.Note):
        self.managed_notes.append(note)

    def transposition_notes(self):
        for note in reversed(self.managed_notes):
            try:
                note.end = self._note_dict[note.pitch]
            except KeyError:
                note.end = max(self.end, note.end)
            self._note_dict[note.pitch] = note.start


# Divided note by note_on, note_off
class SplitNote:
    def __init__(self, type, time, value, velocity):
        ## type: note_on, note_off
        self.type = type
        self.time = time
        self.velocity = velocity
        self.value = value

    def __repr__(self):
        return "<[SNote] time: {} type: {}, value: {}, velocity: {}>".format(
            self.time, self.type, self.value, self.velocity
        )


class Event:
    def __init__(self, event_type, value):
        self.type = event_type
        self.value = value

    def __repr__(self):
        return "<Event type: {}, value: {}>".format(self.type, self.value)

    def to_int(self):
        return START_IDX[self.type] + self.value

    def to_token(self):
        return "<{},{}>".format(self.type, self.value)

    @staticmethod
    def from_int(int_value):
        info = Event._type_check(int_value)
        return Event(info["type"], info["value"])

    @staticmethod
    def from_token(str_value: str):
        info = str_value[1:-1].split(",")
        return Event(info[0], int(info[1]))

    @staticmethod
    def _type_check(int_value):
        range_note_on = range(0, RANGE_NOTE_ON)
        range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON + RANGE_NOTE_OFF)
        range_time_shift = range(
            RANGE_NOTE_ON + RANGE_NOTE_OFF,
            RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,
        )

        valid_value = int_value

        if int_value in range_note_on:
            return {"type": "note_on", "value": valid_value}
        elif int_value in range_note_off:
            valid_value -= RANGE_NOTE_ON
            return {"type": "note_off", "value": valid_value}
        elif int_value in range_time_shift:
            valid_value -= RANGE_NOTE_ON + RANGE_NOTE_OFF
            return {"type": "time_shift", "value": valid_value}
        else:
            valid_value -= RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
            return {"type": "velocity", "value": valid_value}


def _divide_note(notes):
    result_array = []
    notes.sort(key=lambda x: x.start)

    for note in notes:
        on = SplitNote("note_on", note.start, note.pitch, note.velocity)
        off = SplitNote("note_off", note.end, note.pitch, None)
        result_array += [on, off]
    return result_array


def _merge_note(snote_sequence):
    note_on_dict = {}
    result_array = []

    for snote in snote_sequence:
        # print(note_on_dict)
        if snote.type == "note_on":
            note_on_dict[snote.value] = snote
        elif snote.type == "note_off":
            try:
                on = note_on_dict[snote.value]
                off = snote
                if off.time - on.time == 0:
                    continue
                result = pretty_midi.Note(on.velocity, snote.value, on.time, off.time)
                result_array.append(result)
            except:
                print("info removed pitch: {}".format(snote.value))
    return result_array


def _snote2events(snote: SplitNote, prev_vel: int):
    result = []
    if snote.velocity is not None:
        modified_velocity = snote.velocity // 4
        if prev_vel != modified_velocity:
            result.append(Event(event_type="velocity", value=modified_velocity))
    result.append(Event(event_type=snote.type, value=snote.value))
    return result


def _event_seq2snote_seq(event_sequence):
    timeline = 0
    velocity = 0
    snote_seq = []

    for event in event_sequence:
        if event.type == "time_shift":
            timeline += (event.value + 1) / 100
        if event.type == "velocity":
            velocity = event.value * 4
        else:
            snote = SplitNote(event.type, timeline, event.value, velocity)
            snote_seq.append(snote)
    return snote_seq


def _make_time_sift_events(prev_time, post_time):
    time_interval = int(round((post_time - prev_time) * 100))
    results = []
    while time_interval >= RANGE_TIME_SHIFT:
        results.append(Event(event_type="time_shift", value=RANGE_TIME_SHIFT - 1))
        time_interval -= RANGE_TIME_SHIFT
    if time_interval == 0:
        return results
    else:
        return results + [Event(event_type="time_shift", value=time_interval - 1)]


def _control_preprocess(ctrl_changes):
    sustains = []

    manager = None
    for ctrl in ctrl_changes:
        if ctrl.value >= 64 and manager is None:
            # sustain down
            manager = SustainDownManager(start=ctrl.time, end=None)
        elif ctrl.value < 64 and manager is not None:
            # sustain up
            manager.end = ctrl.time
            sustains.append(manager)
            manager = None
        elif ctrl.value < 64 and len(sustains) > 0:
            sustains[-1].end = ctrl.time
    return sustains


def _note_preprocess(susteins, notes):
    note_stream = []
    if len(susteins) == 0:
        return notes

    for sustain in susteins:
        for note_idx, note in enumerate(notes):
            if note.start < sustain.start:
                note_stream.append(note)
            elif note.start > sustain.end:
                notes = notes[note_idx:]
                sustain.transposition_notes()
                break
            else:
                sustain.add_managed_note(note)

    for sustain in susteins:
        note_stream += sustain.managed_notes

    note_stream.sort(key=lambda x: x.start)
    return note_stream


def encode_midi(file_path) -> list:
    events = []
    notes = []
    mid = pretty_midi.PrettyMIDI(midi_file=file_path)

    for inst in mid.instruments:
        if inst.program not in [0, 1]:
            continue
        inst_notes = inst.notes
        # ctrl.number is the number of sustain control. If you want to know abour the number type of control,
        # see https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
        ctrls = _control_preprocess(
            [ctrl for ctrl in inst.control_changes if ctrl.number == 64]
        )
        notes += _note_preprocess(ctrls, inst_notes)

    dnotes = _divide_note(notes)

    # print(dnotes)
    dnotes.sort(key=lambda x: x.time)
    # print('sorted:')
    # print(dnotes)
    cur_time = 0
    cur_vel = 0
    for snote in dnotes:
        events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)
        events += _snote2events(snote=snote, prev_vel=cur_vel)
        # events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)

        cur_time = snote.time
        cur_vel = snote.velocity

    return [e.to_token() for e in events]


def decode_midi(idx_array, developer="Pzzzzz", file_path=None):
    event_sequence = [
        Event.from_token(idx) if not isinstance(idx, Event) else idx
        for idx in idx_array
        if "unk" not in idx and "pad" not in idx and idx != ""
    ]
    # print(event_sequence)
    snote_seq = _event_seq2snote_seq(event_sequence)
    note_seq = _merge_note(snote_seq)
    note_seq.sort(key=lambda x: x.start)

    mid = pretty_midi.PrettyMIDI()
    # if want to change instument, see https://www.midi.org/specifications/item/gm-level-1-sound-set
    instument = pretty_midi.Instrument(1, False, "Developed By {}".format(developer))
    instument.notes = note_seq

    mid.instruments.append(instument)
    if file_path is not None:
        mid.write(file_path)
    return mid


maestro_json = None
cnt = 0


def encode_single_worker_legacy(args, path):
    ls = []
    for file in os.listdir(path):
        try:
            whole_seq = encode_midi(os.path.join(path, file))
        except Exception as e:
            print("Problematic file {}".format(file))
            continue
        ls.append(whole_seq)
        """
        for bg in range(0, max(len(whole_seq) - 2048, 1), 2048):
            ls.append(whole_seq[bg : bg + 2048])
        """
        """
        whole_seq = split_sequence(whole_seq, args.maxlen)
        for seg in whole_seq:
            ls.append(seg)
        """
    return ls


def encode_single_worker(args, Theard_ind, jobs: list) -> list:
    res = []
    for job in tqdm(jobs):
        if args.other:
            whole_seq = encode_midi(job)
        else:
            whole_seq = encode_midi(
                os.path.join(args.datadir, "maestro-v2.0.0", job["midi_filename"])
            )
            whole_seq = [job["midi_filename"]] + whole_seq
        res.append(whole_seq)
    return res, Theard_ind


def remi_encode_single_worker(args, jobs: list):
    def encode_remi(path):
        def extract_events(input_path):
            import utils

            note_items, tempo_items = utils.read_items(input_path)
            note_items = utils.quantize_items(note_items)
            max_time = note_items[-1].end
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
            groups = utils.group_items(items, max_time)
            events = utils.item2event(groups)
            return events

        midi_paths = [path]
        all_events = []
        for path in midi_paths:
            events = extract_events(path)
            all_events.append(events)
        # event to word
        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = "{}_{}".format(event.name, event.value)
                words.append(e.replace(" ", "_"))
            all_words.append(words)
        return all_words[0]

    res = []
    for job in tqdm(jobs):
        whole_seq = encode_remi(
            os.path.join(args.datadir, "mae-remove-sustain", job["midi_filename"])
        )
        whole_seq = [job["split"]] + whole_seq
        res.append(whole_seq)
    return res


def split_sequence(notes, max_length):
    on_ls = []
    res = []
    seg = []
    time_shift = []
    log_on = []
    for note in notes:
        note = Event.from_token(note)
        time_shift.append(note)
        if "note_on" == note.type:
            on_ls.append(note.value)
        elif "note_off" == note.type:
            assert note.value in on_ls
            on_ls.remove(note.value)
        elif "time_shift" == note.type:
            if len(seg) + len(log_on) + len(time_shift) > max_length:
                tmp = seg[-1]
                seg.pop()
                seg += [Event("note_off", i) for i in log_on]
                seg.append(tmp)
                res.append(seg)
                seg = []
                time_shift = [Event("note_on", i) for i in log_on] + time_shift
            else:
                seg += time_shift
                time_shift = []
            log_on = [i for i in on_ls]
    res.append(seg + time_shift)
    for ind, seg in enumerate(res):
        res[ind] = [i.to_token() for i in seg]
    return res


def main(args):
    os.makedirs(args.destdir, exist_ok=True)

    if args.other:
        midi_files = glob(os.path.join(args.datadir, "*.mid"))
        midi_files += glob(os.path.join(args.datadir, "*.midi"))
        assert len(midi_files) != 0, "No midi file found under path {}".format(
            args.datadir
        )

        total = len(midi_files)
        train = 8 * total // 10
        valid = total // 10
        test = total - train - valid
        global cnt
        cnt = 0

        if args.prefix == None:
            prefix = os.path.split(args.datadir)[-1]
        else:
            prefix = args.prefix

        print(
            "Total {} midi files, split to {} train sample(s) | {} test sample(s) | {} train sample(s)\nPrefix: {}".format(
                total, train, test, valid, prefix
            )
        )
        try:
            for sp in ["test", "train", "valid"]:
                os.remove(os.path.join(args.destdir, f"{prefix}.{sp}.tokens"))
        except FileNotFoundError:
            pass

        pool = Pool(args.workers)

        def merge_results(worker_result):
            global cnt
            worker_result, Thread_ind = worker_result
            with open(
                os.path.join(args.destdir, f"{prefix}.test.tokens"), "a+"
            ) as tst, open(
                os.path.join(args.destdir, f"{prefix}.train.tokens"), "a+"
            ) as tr, open(
                os.path.join(args.destdir, f"{prefix}.valid.tokens"), "a+"
            ) as vd:
                for line in worker_result:
                    line = " ".join(line)
                    line += "\n"
                    if cnt < train:
                        tr.write(line)
                    elif cnt < train + test:
                        tst.write(line)
                    else:
                        vd.write(line)
                    cnt += 1
            print(f"Thread {Thread_ind} finished. ")

        step = (total + args.workers - 1) // args.workers
        for ind in range(0, total, step):
            result = pool.apply_async(
                encode_single_worker,
                (args, ind, midi_files[ind : ind + step]),
                callback=merge_results,
            )
        result.get()
        pool.close()
        pool.join()

    else:
        global maestro_json
        maestro_json = json.load(
            open(os.path.join(args.datadir, "maestro-v2.0.0.json"), "r")
        )

        try:
            for sp in ["test", "train", "valid"]:
                os.remove(os.path.join(args.destdir, f"mae_remi.{sp}.tokens"))
        except FileNotFoundError:
            pass

        pool = Pool(args.workers)

        def merge_results(worker_result):
            worker_result, Thread_ind = worker_result
            global maestro_json
            with open(args.destdir + "/mae_remi.test.tokens", "a+") as tst, open(
                args.destdir + "/mae_remi.train.tokens", "a+"
            ) as tr, open(args.destdir + "/mae_remi.valid.tokens", "a+") as vd:
                for line in worker_result:
                    split, line = line[0], " ".join(line[1:])
                    if len(line.strip().split()) < 2048:
                        continue
                    line += "\n"
                    if split == "train":
                        tr.write(line)
                    elif split == "test":
                        tst.write(line)
                    else:
                        vd.write(line)
            print(f"{Thread_ind} finished. ")

        seg = (len(maestro_json) + args.workers - 1) // args.workers
        ind = 0

        for i in range(args.workers):
            if i == args.workers - 1:
                seg = len(maestro_json)
            result = pool.apply_async(
                encode_single_worker,
                (args, i, maestro_json[ind : ind + seg]),
                callback=merge_results,
            )
            print(ind, ind + seg)
            ind += seg
        result.get()
        pool.close()
        pool.join()


def cli_main():
    parser = argparse.ArgumentParser("midi preprocess parser")

    parser.add_argument(
        "--destdir", metavar="DIR", default="./", help="destination dir"
    )
    parser.add_argument(
        "--other", action="store_true", default=False, help="using maestro or not"
    )
    parser.add_argument("--datadir", metavar="DIR", default="data", help="data dir")
    parser.add_argument("--workers", type=int, default="20")
    parser.add_argument("--prefix", type=str)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
