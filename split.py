#!/home/zhangyi/miniconda3/envs/fs/bin/python

import sys
from subprocess import Popen, PIPE

toascii = '/home/zhangyi/midifile/bin/toascii'
tobinary = '/home/zhangyi/midifile/bin/tobinary'

def writeAsc(segNum, common, seg, highlight=False):
    filename = '%s_%d.mid' % (sys.argv[1].rsplit('.', 1)[0], segNum)
    if highlight: filename = filename.replace('.mid', '_HIGH.mid')
    with Popen([tobinary, filename], stdin=PIPE) as proc:
        f = proc.stdin
        for line in common+seg: f.write(bytes(line, 'ASCII'))
        f.write(b'v0\tff 2f v0\n') # End of track
    print('Wrote', len(seg), 'items to', filename)

if len(sys.argv) < 2:
    print('Usage: splitasc.py <midifile.mid> [splitlimit]')
    print('\nSplit limit is in seconds, default is 5')
    exit(1)

if len(sys.argv) > 2:
    limit = int(sys.argv[2])
else:
    limit = 5

with Popen([toascii, sys.argv[1]], stdout=PIPE) as proc:
    fin = proc.stdout
    T = 0 # current time base
    pedals = set() # active pedals
    keys = set() # active keys
    silent = True # Is all silent?
    silentT = 0 # Start of silent

    common = [] # Common lines in beginning
    seg = [] # Current segment
    segNum = 1 # Running number

    pedCount = 0 # End with double press of pedal to highlight

    for line in fin:
        line = line.decode('ASCII')
        if len(line) == 0 or line[0] != 'v':
            common.append(line)
            if len(common) == 5: # Try to extract PPQ
                try: PPQ = int(line[2:])
                except: pass
            continue

        it = line.split()
        T += int(it[0][1:])

        if it[1] == 'ff':
            if it[2] != '2f': # Skip EOF
                common.append(line) # Tempo and time etc. into common
                if it[2] == '51': # Try to extract BPM
                    try: BPM = int(it[-1][1:])
                    except: pass
            continue

        if it[1] == 'b0': # Controller info
            if it[3] == "'0":
                pedals.discard(it[2])
            else:
                if not pedals: pedCount += 1 # Increase pedal count
                pedals.add(it[2])
        elif it[1] == '90': # Note on
            keys.add(it[2])
        elif it[1] == '80': # Note off
            keys.discard(it[2])

        if (pedals or keys) and silent: # End of silence
            silent = False
            if T-silentT > limit*PPQ*BPM/60: # Over specified limit
                line = 'v0\t%s\n' % (' '.join(it[1:])) # hack zero timedelta
                if seg: # Data to write
                    writeAsc(segNum, common, seg, pedCount > 1)
                    segNum += 1
                seg = [] # Clear segment
        else: silent, silentT = True, T

        seg.append(line)
        if it[1] == '90': pedCount = 0 # Delayed reset of pedCount on notes

    if seg: # Data to write
        writeAsc(segNum, common, seg, pedCount > 1)
        segNum += 1