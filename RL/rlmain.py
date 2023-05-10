import numpy as np
import re
import pickle
import sys
import colorama
from colorama import Fore, Style

class State:
    def __init__(self,blen,p1,p2,debug=False):
        self.len = blen
        self.p1 = p1
        self.p2 = p2
        self.board = np.array((' '.join('x'*5+'_'*(self.len-10)+'o'*5)).split(' '))
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1
        self.debug = debug

    def getHash(self):
        self.boardHash = ''.join(self.board)
        return self.boardHash

    def winner(self):
        if(len(np.argwhere(self.board == 'x')) <2):
            self.isEnd = True
            return -1
        if(len(np.argwhere(self.board == 'o')) <2):
            self.isEnd = True
            return 1
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        if (self.playerSymbol == 1):
            r_idx = np.argwhere(self.board=='x').reshape(1,-1)[0]
            b_idx = np.argwhere(self.board=='_').reshape(1,-1)[0]
            positions = np.array(np.meshgrid(r_idx,b_idx)).T.reshape(-1,2)
        else:
            r_idx = np.argwhere(self.board=='o').reshape(1,-1)[0]
            b_idx = np.argwhere(self.board=='_').reshape(1,-1)[0]
            positions = np.array(np.meshgrid(r_idx,b_idx)).T.reshape(-1,2)
        return positions

    def validate_board(self):
        opponent = ''
        if (self.playerSymbol == 1):
            kill = re.compile('(?=(xo+x))')
            suicide = re.compile('(?=(ox+o))')
            opponent = 'o'
        else:
            kill = re.compile('(?=(ox+o))')
            suicide = re.compile('(?=(xo+x))')
            opponent = 'x'

        s = ''.join(self.board)
        for m in re.finditer(kill,s):
            if(self.debug):
                print(f'board_validate: Detected kill at {m.start(1)}:{m.end(1)}')
            self.board[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')

        s = ''.join(self.board)
        for m in re.finditer(suicide,s):
            if(self.debug):
                print(f'board_validate: Detected suicide at {m.start(1)}:{m.end(1)}')
            self.board[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')

        # Switch the symbol for the next player to play
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1
        return

    def updateState(self,position):
        printstr = ''
        if(self.debug):
            printstr = f'{self.board} {self.playerSymbol}:[{position[0]}->{position[1]}'
        self.board[[position[0],position[1]]]= self.board[[position[1],position[0]]]
        if(self.debug):
            print(printstr+f'{self.board}')

        return

    def giveReward(self):
        result = self.winner()
        if result == 1:
            if(self.debug):
                print('Detected Victory for x')
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        else:
            if(self.debug):
                print('Detected Victory for o')
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        return

    def reset(self):
        self.board = np.array((' '.join('x'*5+'_'*(self.len-10)+'o'*5)).split(' '))
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1
        return

    def play (self,rounds=100):
        for i in range(rounds):
            if(self.debug):
                print(f'Playing game {i}')
            if i%1000 == 0:
                print(f'Rounds {i}')
            while not self.isEnd:
                # player 1 - x 
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions,self.board,self.playerSymbol)
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                self.validate_board()
                
                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
                else:
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions,self.board,self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)
                    self.validate_board()

                    win = self.winner()
                    if win is not None:
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
    
    def play2 (self):
        self.showBoard()
        while not self.isEnd:
            positions = self.availablePositions()
            if (self.p1.name == "computer"):
                p1_action = self.p1.chooseAction(positions,self.board,self.playerSymbol)
            else:
                p1_action = self.p1.chooseAction(positions)

            self.updateState(p1_action)
            self.validate_board()
            self.showBoard()
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print(self.p2.name, "wins!")
                self.reset()
                break
            else:
                positions = self.availablePositions()
                if (self.p2.name == "computer"):
                    p2_action = self.p2.chooseAction(positions,self.board,self.playerSymbol)
                else:
                    p2_action = self.p2.chooseAction(positions)
                self.updateState(p2_action)
                self.validate_board()
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print(self.p1.name, "wins!")
                    self.reset()
                    break
        return

    def showBoard(self):
        print(''.join(self.board))
        print("0123456789abcd")
        return


class Player:
    def __init__ (self,name,exp_rate=0.3):
        self.name = name
        self.states = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}
    
    def getHash(self,board):
        return ''.join(board)

    def chooseAction(self,positions,current_board,symbol):
        if np.random.uniform(0,1) <= self.exp_rate:
            action = positions[np.random.choice(len(positions))]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[[p[0],p[1]]] = next_board[[p[1],p[0]]]
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    def addState(self,state):
        self.states.append(state)
        return

    def feedReward(self,reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr*(self.decay_gamma*reward-self.states_value[st])
            reward = self.states_value[st]
        return

    def reset(self):
        self.states = []
        return
    
    def savePolicy(self):
        fw = open('policy_'+str(self.name),'wb')
        pickle.dump(self.states_value,fw)
        fw.close()
        return

    def loadPolicy(self,file):
        fr = open(file,'rb')
        self.states_value = pickle.load(fr)
        fr.close()

class HumanPlayer:
    def __init__(self,name):
        self.name = name
        return
    
    def chooseAction(self,positions):
        while True:
            rock = int(input(Fore.BLUE+"From Location: "))
            blank = int(input(Fore.BLUE+"To Location: "))
            print(Style.RESET_ALL)
            if np.isin(rock,positions[:,0]) and np.isin(blank,positions[:,1]):
                return (rock,blank)
            else:
                print(Fore.RED+"Wrong input. Try again..")
                continue
        return

    def addState (self,state):
        pass
    def feedReward(self,reward):
        pass
    def reset(self):
        pass

if __name__ == "__main__":
    if (len(sys.argv) == 2):
        if (sys.argv[1] == 'train'):
            p1 = Player("x")
            p2 = Player("o")

            st = State(14,p1,p2)
            st.play(100000)

            p1.savePolicy()
            p2.savePolicy()
        elif (sys.argv[1] == 'play'):
            toss = np.random.randint(2)
            if (toss):
                print("You won the toss .. Make the first move")
                p2 = Player ("computer", exp_rate =0)
                p2.loadPolicy("policy_o")
                p1 = HumanPlayer("human")
            else:
                print("Computer won the toss .. Will make the first move")
                p1 = Player ("computer", exp_rate =0)
                p1.loadPolicy("policy_x")
                p2 = HumanPlayer("human")

            st = State(14,p1,p2)
            st.play2()
        else:
            print('Usage : python rlmain.py [train | play]')
    else:
        print('Usage : python rlmain.py [train | play]')
        
