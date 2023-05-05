import numpy as np
import pickle
import sys
import re
import random
from pathlib import Path

class Board:
    __blen = 10
    __barr = np.array(['x','x','x','x','_','_','o','o','o','o'])
    __offside = 'x'
    __lastplayed = 'x'
    __debug=True
    __kholoc = []
    __rounds = np.array([])
    __outcome = np.array([])
    __suicide = False
    def __init__(self,boardlen=10,os = 'x',debug=False):
        self.blen = boardlen
        while ((self.blen <= 10) | (self.blen > 30)):
            self.blen = int(input ("A good board length is between 11 and 30: "))
        self.barr = np.array((' '.join('x'*5+'_'*(self.blen-10)+'o'*5)).split(' '))
        self.offside = os
        self.debug=debug
        self.kholoc = []
        self.rounds = np.array([])
        self.outcome = np.array([])
        self.suicide = False
        return

    def get_board(self,sep=''):
        return sep.join(self.barr)

    def random_play(self,offside):
        self.last_played=offside
        # find the indices where we have the offensive side rocks are
        c_idx = np.argwhere(self.barr == offside).reshape(1,-1)[0]
        # find the blank indices
        b_idx = np.argwhere(self.barr == '_').reshape(1,-1)[0]
        # Now, remove kholocations
        #b_idx = np.delete(b_idx, np.isin(b_idx,self.kholoc)) 
        if(len(b_idx) == 0):
            raise Exception("random_play: nowhere to place the rock!")
        randrock = random.choice(c_idx)
        randplace = random.choice(b_idx)
        printstr = ''
        if(self.debug):
            printstr += f'offside is {offside}: board is [{self.barr}]: {randrock}->{randplace} ->newboard '
        prevarr = np.array(self.barr)
        self.barr[[randrock,randplace]] = self.barr[[randplace,randrock]]
        if (len(self.rounds)):
            self.rounds = np.vstack((self.rounds,np.concatenate((prevarr,self.barr))))
        else:
            self.rounds = np.concatenate((prevarr,self.barr))

        if(self.debug):
            printstr += f'[{self.barr}]'
            print(printstr)
        return

    def encode(self,rdata,offside):
        retdata = np.array(rdata)
        retdata = np.where(retdata=='_','0',retdata)
        if (offside == 'x'):
            retdata = np.where(retdata=='x','1',retdata)
            retdata = np.where(retdata=='o','-1',retdata)
        else:
            retdata = np.where(retdata=='x','-1',retdata)
            retdata = np.where(retdata=='o','1',retdata)
        return retdata.astype(int)    

    def createlabel (self):
        if(self.last_played == 'x'):
            # We know 'x' won and it played first
            labelstr = '10'*(int((len(self.rounds)+1)/2))
            self.outcome = np.array((' '.join(labelstr)).split(' '))[0:-1]
            self.rounds = self.encode(self.rounds,'x')
            if(self.debug):
                print(self.outcome,self.outcome.shape)
        else:
            # We know 'o' won and it played second 
            labelstr = '01'*(int(len(self.rounds)/2))
            self.outcome = np.array((' '.join(labelstr)).split(' '))
            self.rounds = self.encode(self.rounds,'o')
            if (self.debug):
                print(self.outcome,self.outcome.shape)
        if (self.debug):
            print(self.rounds.shape,type(self.rounds),type(self.outcome))
        if (len(self.rounds) < 30):
            if Path("./alak_x.pkl").is_file():
                rounds = np.array([])
                outcome = np.array([])
                with open("./alak_x.pkl","rb") as rfile:
                    rounds = pickle.load(rfile)
                    rounds = np.vstack((rounds,self.rounds))
                with open("./alak_y.pkl","rb") as rfile:
                    outcome = pickle.load(rfile)
                    outcome = np.concatenate((outcome,self.outcome.astype(int)))
                with open("./alak_x.pkl","wb") as wfile:
                    pickle.dump(rounds,wfile)
                with open("./alak_y.pkl","wb") as wfile:
                    pickle.dump(outcome,wfile)
            else:
                with open("./alak_x.pkl","wb") as wfile:
                    pickle.dump(self.rounds,wfile)
                with open("./alak_y.pkl","wb") as wfile:
                    pickle.dump(self.outcome.astype(int),wfile)
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
            if((m.end(1)-m.start(1)) == 3):
                # We got a kill here
                self.kholoc.append(m.start(1)+1)
        s = ''.join(self.barr)
        for m in re.finditer(suicide,s):
            if(self.debug):
                print(f'board_validate: Detected suicide at {m.start(1)}:{m.end(1)}')
            self.barr[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
            self.suicide = True
            return False
            
        if(len(np.argwhere(self.barr == opponent)) >1):
            return True
        else:
            return False
    
    def suicide_detected(self):
        return self.suicide


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
        elif (sys.argv[1] == "generate"):
            no_games = 60000
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
                        if(b.suicide_detected()):
                            print("suicide detected. Discarding game")
                            break
                        print(f'x wins the game: {i+1}')
                        b.createlabel()
                        xwins +=1
                        break
                    b.random_play('o')
                    game_on = b.board_validate()
                    if(not game_on):
                        if(b.suicide_detected()):
                            print("suicide detected. Discarding game")
                            break
                        print(f'o wins the game: {i+1}')
                        b.createlabel()
                        owins +=1
                        break
            print(f'x wins {xwins} games, o wins {owins}games. Total played: {no_games}')
    else:
        print("Usage:\npython newmain.py [test|generate]")

