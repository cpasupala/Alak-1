import random
import re
import numpy as np
import colorama
from colorama import Fore, Style
import pickle
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class NNClass:
    model = MLPClassifier
    nnrock = 'x'
    def __init__(self,rock):
        self.nnrock = rock
        if Path("./perfect_model.pkl").is_file():
            self.model = pickle.load(open("./perfect_model.pkl","rb"))
        else:
            raise Exception("Could not find the model. Exiting")
        return

    def nnplay (self,board,rock,kholocs):
        r_idx = np.argwhere(board==rock).reshape(1,-1)[0]
        if (len(r_idx)==0):
            raise Exception(Fore.RED+"There are no rocks and still playing. This should  not happen!")

        b_idx = np.argwhere(board=='_').reshape(1,-1)[0]
        if (len(b_idx)==0):
            raise Exception(Fore.RED+"No place on the board ? You must be kidding me!")
        X = []
        rx = []
        for r in r_idx:
            for x in b_idx:
                temp_board = np.array(board)
                temp_board[[r,x]] = temp_board[[x,r]]
                if(not detect_suicide(temp_board,rock)):
                    rx.append((r,x))
                    X.append(np.concatenate((board,temp_board)))
        X = np.array(X)

        opponent = 'x'
        if (rock == 'x') :
            opponent = 'o'
        else:
            opponent = 'x'

        X = np.where(X==rock,1, X)
        X = np.where(X=='_',0,X)
        X = np.where(X==opponent,-1,X)
        X = np.array(X,dtype=float)

        probarray = self.model.predict_proba(X)[:,1]
        return (rx[np.argmax(probarray)])
        

class Board:

    blen = 14
    barr = []
    nn_rock ='x'
    rand_rock = 'o'
    err_code = 0
    kholoc = []
    debug = False
    
    def __init__(self,blen,toss):
        self.blen = blen
        if (toss):
            nn_rock = 'x'
            rand_rock = 'o'
        else:
            nn_rock = 'o'
            rand_rock = 'x'
        self.barr = np.array((' '.join('x'*5+'_'*(blen-10)+'o'*5)).split(' '))
        self.nn_rock = 'x'
        self.rand_rock = 'o'
        self.err_code = 0
        self.kholoc = []
        self.debug = False
        return

    def detect_suicide(self, board, rock):
        s = ''.join(board)
        if (rock == 'x'):
            kill = re.compile('(?=(xo+x))')
            suicide = re.compile('(?=(ox+o))')
        else:
            kill = re.compile('(?=(ox+o))')
            suicide = re.compile('(?=(xo+x))')

        for m in re.finditer(kill,s):
            board[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
        s = ''.join(board)
        for m in re.finditer(suicide,s):
            board[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
            return True

        return False

    def validate(self,rock):
        self.kholoc = []
        opponent = ''
        gain = 0
        if (rock == 'x'):
        # Parse for the kill and then for suicide
            kill = re.compile('(?=(xo+x))')
            suicide = re.compile('(?=(ox+o))')
            opponent = 'o'
        else:
            kill = re.compile('(?=(ox+o))')
            suicide = re.compile('(?=(xo+x))')
            opponent = 'x'

        s = ''.join(self.barr)
        for m in re.finditer(kill,s):
            if(self.debug):
                print(f'board_validate: Detected kill at {m.start(1)}:{m.end(1)}')
            self.barr[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
            gain += m.end(1)-m.start(1)-4
            if((m.end(1)-m.start(1)) == 5):
                # We got a kill here
                self.kholoc.append(m.start(1)+1)
        s = ''.join(self.barr)
        for m in re.finditer(suicide,s):
            if(self.debug):
                print(f'board_validate: Detected suicide at {m.start(1)}:{m.end(1)}')
            self.barr[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
            gain -= m.end(1)-m.start(1)-4
            
        if(len(np.argwhere(self.barr == opponent)) >1):
            return True
        else:
            return False

    def play_n_validate(self,playfunc,rock):
        '''
        Get all possible valid moves for the rock. call the function given by 
        the user. The function 'playfunc' should return the from-to combination
        of indices.
        board verifies if it is a legal move and then updates the board.
        '''
        r_idx = np.argwhere(self.barr==rock).reshape(1,-1)[0]
        if (len(r_idx)==0):
            raise Exception(Fore.RED+"There are no rocks and still playing. Bad Kitty!")
            self.err_code = -1
            return False

        b_idx = np.argwhere(self.barr=='_').reshape(1,-1)[0]
        if (len(b_idx)==0):
            raise Exception(Fore.RED+"No place on the board ? You must be kidding me!")
            self.err_code = -2
            return False
        
        '''
        Kho is allowed. So, we dont exclude those locations. That is the 
        reason for commenting out this code.

        # Exclude kho locations
        b_idx = np.delete(b_idx, np.isin(b_idx,self.kholoc))
        if (len(b_idx)==0):
            print(Fore.RED+"No place on the board after remvoing Kho locations? ")
            self.err_code = -3
            return False
        '''
        
        randrock = random.choice(r_idx)
        suicide_list = []
        for randrock in r_idx:
            for x in b_idx:
                temp_board = np.array(self.barr)
                temp_board[[randrock,x]] = temp_board[[x,randrock]]
                if(self.detect_suicide(temp_board,rock)):
                    suicide_list.append((randrock,x))

        if(self.debug):
            print(f'legal moves: {b_idx}')
            print(f'suicide moves: {suicide_list}')
        # Now, we know all the legal moves and its subset non-suicide legal moves
        # See what the user function returns
        u_rock,u_place = playfunc(np.array(self.barr),rock,self.kholoc)
        if(self.debug):
            print(f'{rock} moved {u_rock}->{u_place}')

        if(np.isin(u_rock,r_idx) and np.isin(u_place,b_idx)):
            if (self.debug):
                bstr = ''.join(self.barr)
                printstr = f'{bstr} [{rock}:{u_rock}->{u_place}] '
            self.barr[[u_rock,u_place]] = self.barr[[u_place,u_rock]]
            if(self.debug):
                bstr = ''.join(self.barr)
                print(printstr+f'{bstr}')
            if(len(suicide_list)>0):
                if ((u_rock,u_place) in suicide_list):
                    print(Fore.RED+f'Warning: Player {rock} made a suicide move. Ignoring')
                    print(Style.RESET_ALL)
            # Clear Board and check for victory !!
            if(self.validate(rock)):
                return True
            else:
                self.err_code = -3
                return False
        else:
            raise Exception (Fore.RED+f'Error: Player {rock} played an illegal move')
            print(self.barr)
            print(f'{rock}:{u_rock}-->{u_place}')
            self.err_code = -4
            return False

manual = False
def randplay(board,rock,kholocs):
    if(manual):
        print(board)
        r_idx = int(input("Which Rock: "))
        b_idx = int(input("Where to  : "))
        return r_idx,b_idx
        
    r_idx = np.argwhere(board==rock).reshape(1,-1)[0]
    b_idx = np.argwhere(board=='_').reshape(1,-1)[0]
    #b_idx = np.delete(b_idx, np.isin(b_idx,kholocs))
    return random.choice(r_idx), random.choice(b_idx)

def detect_suicide(board, rock):
    s = ''.join(board)
    if (rock == 'x'):
        kill = re.compile('(?=(xo+x))')
        suicide = re.compile('(?=(ox+o))')
    else:
        kill = re.compile('(?=(ox+o))')
        suicide = re.compile('(?=(xo+x))')

    for m in re.finditer(kill,s):
        board[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
    s = ''.join(board)
    for m in re.finditer(suicide,s):
        board[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
        return True

    return False



if __name__ == "__main__":

    # Toss
    no_games = 100
    ANN_vict_stat = []
    ann_vict = 0
    for games in range(no_games):
        game_on = True
        toss = np.random.randint(2)
        if(toss):
            nn_rock = 'x'
            rand_rock = 'o'
            #print(f'Game No {games}: NN won the toss.')
        else:
            nn_rock = 'o'
            rand_rock = 'x'
            #print(f'Game No {games}: Rand won the toss.')

        # setup Board 
        b = Board(14,toss)

        if(toss == 0):
            game_on = b.play_n_validate(randplay,rand_rock)
            if (not game_on):
                raise Exception("Game cant be over in one move!!")

        nnmodel = NNClass(nn_rock)

        altfuncs =[nnmodel.nnplay,randplay]
        rocks = [nn_rock,rand_rock]
        i = 0
        while (game_on):
            game_on = b.play_n_validate(altfuncs[i%2],rocks[i%2])
            i += 1
        if(i%2):
            ANN_vict_stat.append([nn_rock,1,i+1])
            ann_vict += 1
            #print(f'NN Won in {i} steps!!')
        else:
            ANN_vict_stat.append([nn_rock,0,i+1])
            #print(f'RANDPLAY Won in {i} steps!!')

    #print(np.array(ANN_vict_stat))
    print(f'NN won {ann_vict} out of {no_games} games')
