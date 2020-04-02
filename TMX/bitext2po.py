# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:15:24 2018

@author: ruben
"""
import sys
import os, glob, re

if len(sys.argv) < 3:  # need to input at least 1 arguments
    print(f'Usage: {sys.argv[0]} input.en input.zh\n', file=sys.stderr)
    sys.exit()

fi1n = sys.argv[1]
if not os.path.exists(fi1n):
    print(f"File '{fi1n}' does not exist!", file=sys.stderr)
    sys.exit()
fi2n = sys.argv[2]
if not os.path.exists(fi2n):
    print(f"File '{fi2n}' does not exist!", file=sys.stderr)
    sys.exit()

basen = os.path.basename(fi1n)
(file, ext) = os.path.splitext(basen)
print(f"Base file name = {file}")


fon = f"{file}.po" 
fo  = open(fon, "w", encoding="utf-8", newline="\n")

n = 0
with open(fi1n, "rt", encoding="utf-8") as f1i:
    with open(fi2n, "rt", encoding="utf-8") as f2i:
        for line1 in f1i:
            line1 = line1.strip()
            line2 = next(f2i)
            line2 = line2.strip()
            n += 1
            fo.write(f'msgid "{line1}"\n')
            fo.write(f'msgstr "{line2}"\n')
            fo.write("\n")

fo.close()
