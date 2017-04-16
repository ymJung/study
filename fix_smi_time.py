import codecs
import sys

SEC = 60
MIL_SEC = 1000
if sys.argv[0] is not None:
    file_name = sys.argv[0]
else:
	file_name = 'blah.smi'


min = 0
sec = 0 


fix_time = min * SEC * MIL_SEC + (sec * MIL_SEC)

new_file = open('FIX_' + file_name, 'w', encoding='utf-8', newline='')
prefix = '<SYNC Start='
surfix = '><P Class=ENUSCC'

lines = codecs.open(file_name, 'r', 'utf-8').readlines()
for line in lines:
    if prefix in line :
        origin_pos = line[line.index(prefix) + len(prefix) : line.index(surfix)]
        line = prefix + str(int(origin_pos) + fix_time) + line[line.index(surfix):]
    new_file.write(line)

new_file.close()

