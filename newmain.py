import numpy as np
import sys
import re

class Board:
    __blen = 10
    __barr = np.array(['x','x','x','x','_','_','o','o','o','o'])
    __offside = 'x'
    def __init__(self,boardlen=10,os = 'x'):
        self.blen = boardlen
        while ((self.blen < 10) | (self.blen > 30)):
            self.blen = int(input ("A good board length is between 10 and 30: "))
        self.barr = np.array((' '.join('x'*4+'_'*(self.blen-8)+'o'*4)).split(' '))
        self.offside = os
        return

def validate(s,o):
    sarr = np.array((' '.join(s)).split(' '))
    if (o == 'x'):
    # Parse for the kill and then for suicide
        kill = re.compile('(?=(xo+x))')
        suicide = re.compile('(?=(ox+o))')
    elif (o=='o'):
        kill = re.compile('(?=(ox+o))')
        suicide = re.compile('(?=(xo+x))')
    else:
        raise Exception('Wrong input for offside')

    for m in re.finditer(kill,s):
        sarr[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
    s = ''.join(sarr)
    for m in re.finditer(suicide,s):
        sarr[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
    return ''.join(sarr)

if __name__ == "__main__":
    if(len(sys.argv) == 2):
        if (sys.argv[1] == "test"):
            while(True):
                bstr = input("Input the board(type <end> to end): ")
                if (bstr == 'end'):
                    break
                os = input("Input offside : ")
                print(f'Cleared Board: {validate(bstr,os)}')
    else:
        b = Board(10,'o')
        bstr = ''.join(b.barr)
        print(f'Board initialized to : {bstr}\nThe offensive side   : {b.offside}')

