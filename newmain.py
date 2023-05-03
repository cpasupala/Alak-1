import numpy as np
import sys
import re
import random

class Board:
    __blen = 10
    __barr = np.array(['x','x','x','x','_','_','o','o','o','o'])
    __offside = 'x'
    __lastplayed = 'x'
    __debug=True
    __kholoc = []
    rounds = np.array([])
    def __init__(self,boardlen=10,os = 'x',debug=False):
        self.blen = boardlen
        while ((self.blen <= 10) | (self.blen > 30)):
            self.blen = int(input ("A good board length is between 11 and 30: "))
        self.barr = np.array((' '.join('x'*5+'_'*(self.blen-10)+'o'*5)).split(' '))
        self.offside = os
        self.debug=debug
        self.kholoc = []
        self.rounds = np.array([])
        return

    def get_board(self,sep=''):
        return sep.join(self.barr)

    def random_play(self,offside):
        self.last_played=offside
        # find the indices where we have the offensive side rocks are
        c_idx = np.argwhere(self.barr == offside).reshape(1,-1)[0]
        # find the blank indices
        b_idx = np.argwhere(self.barr == '_').reshape(1,-1)[0]
        if(len(b_idx) == 0):
            raise Exception("random_play: nowhere to place the rock!")
        # Now, remove kholocations
        b_idx = np.delete(b_idx, np.isin(b_idx,self.kholoc)) 
        randrock = random.choice(c_idx)
        randplace = random.choice(b_idx)
        printstr = ''
        if(self.debug):
            printstr += f'offside is {offside}: board is [{self.barr}]: {randrock}->{randplace} ->newboard '
        prevarr = self.barr
        self.barr[[randrock,randplace]] = self.barr[[randplace,randrock]]
        if (len(self.rounds)):
            self.rounds = np.vstack((self.rounds,np.concatenate((prevarr,self.barr))))
        else:
            self.rounds = np.concatenate((prevarr,self.barr))

        if(self.debug):
            printstr += f'[{self.barr}]'
            print(printstr)
        return

    def dump_rounds (self):
        print(self.rounds.shape)
        return

    def board_validate(self):
        self.kholoc = []
        if (self.last_played == 'x'):
        # Parse for the kill and then for suicide
            kill = re.compile('(?=(xo+x))')
            suicide = re.compile('(?=(ox+o))')
            opponent = 'o'
        elif (self.last_played == 'o'):
            kill = re.compile('(?=(ox+o))')
            suicide = re.compile('(?=(xo+x))')
            opponent = 'x'
        else:
            raise Exception('Wrong input for offside')

        s = ''.join(self.barr)
        for m in re.finditer(kill,s):
            if(self.debug):
                print(f'board_validate: Detected kill at {m.start(1)}:{m.end(1)}')
            self.barr[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
            if((m.end(1)-m.start(1)) == 5):
                # We got a kill here
                self.kholoc.append(m.start(1)+1)
        s = ''.join(self.barr)
        for m in re.finditer(suicide,s):
            if(self.debug):
                print(f'board_validate: Detected suicide at {m.start(1)}:{m.end(1)}')
            self.barr[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
        if(len(np.argwhere(self.barr == opponent)) >1):
            return True
        else:
            return False


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
        print(m.start(1),m.end(1))
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
        no_games = 2
        xwins = 0
        owins = 0
        for i in range(no_games):
            b = Board(14,'x')
            board_str = b.get_board(sep=' ')
            #print(f'Board initialized to : {board_str}')
            #print(f'The offensive side   : {b.offside}')

            game_on = True
            while(game_on):
                b.random_play('x')
                game_on = b.board_validate()
                if(not game_on):
                    #print('x wins the game')
                    b.dump_rounds()
                    xwins +=1
                    break
                b.random_play('o')
                game_on = b.board_validate()
                if(not game_on):
                    #print('o wins the game')
                    b.dump_rounds()
                    owins +=1
                    break
        print(f'x wins {(xwins/no_games)*100:.1f}% of {no_games } played')

