import numpy as np
import re
import pickle
import sys
import colorama
from colorama import Fore, Style

class State:
    def __init__(self,blen,p1,p2,debug=False,verbose=True):
        self.len = blen
        self.p1 = p1
        self.p2 = p2
        self.board = np.array((' '.join('x'*5+'_'*(self.len-10)+'o'*5)).split(' '))
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1
        self.debug = debug
        self.inparr = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d']
        self.verbose = verbose

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
        gain = 0
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
            gain += m.end(1) -m.start(1) -2

        s = ''.join(self.board)
        for m in re.finditer(suicide,s):
            if(self.debug):
                print(f'board_validate: Detected suicide at {m.start(1)}:{m.end(1)}')
            self.board[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
            gain -= m.end(1)-m.start(1)-2

        if (self.verbose):
            if (gain >0): 
                print(Fore.GREEN+f'Gain: {gain}')
            elif (gain <0):
                print(Fore.RED+f'Gain: {gain}')
            else:
                print(Fore.YELLOW+f'Gain: {gain}')
            print(Style.RESET_ALL)
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
    
    def play2 (self,verbose=True):
        wonby = ''
        if(verbose):
            self.showBoard()
        gameround = 0
        while not self.isEnd:
            gameround += 1
            if (verbose):
                print(f'Round : {gameround}')
            positions = self.availablePositions()
            if (self.p1.name == "computer"):
                p1_action = self.p1.chooseAction(positions,self.board,self.playerSymbol)
            else:
                p1_action = self.p1.chooseAction(positions)

            if(verbose):
                print(Fore.MAGENTA+f'x moves {self.inparr[p1_action[0]]}->{self.inparr[p1_action[1]]}')
                print(Style.RESET_ALL)

            self.updateState(p1_action)

            if(verbose):
                self.showBoard()

            self.validate_board()

            if(verbose):
                self.showBoard()
            win = self.winner()
            if win is not None:
                if win == 1:
                    wonby = self.p1.name
                    print(self.p1.name, "wins!")
                else:
                    wonby = self.p2.name
                    print(self.p2.name, "wins!")
                self.reset()
                break
            else:
                positions = self.availablePositions()
                if (self.p2.name == "computer"):
                    p2_action = self.p2.chooseAction(positions,self.board,self.playerSymbol)
                else:
                    p2_action = self.p2.chooseAction(positions)
                if (verbose):
                    print(Fore.MAGENTA+f'o moves {self.inparr[p2_action[0]]}->{self.inparr[p2_action[1]]}')
                    print(Style.RESET_ALL)

                self.updateState(p2_action)

                if(verbose):
                    self.showBoard()

                self.validate_board()

                if(verbose):
                    self.showBoard()

                win = self.winner()
                if win is not None:
                    if win == -1:
                        wonby = self.p2.name
                        print(self.p2.name, "wins!")
                    else:
                        wonby = self.p1.name
                        print(self.p1.name, "wins!")
                    self.reset()
                    break
        return wonby

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

class RandomPlayer:
    def __init__(self,name):
        self.name = name
        return

    def chooseAction(self,positions):
        return positions[np.random.randint(len(positions))]

    def addState (self,state):
        pass
    def feedReward(self,reward):
        pass
    def reset(self):
        pass

    
class HumanPlayer:
    def __init__(self,name):
        self.name = name
        self.inpdict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,'b':11,'c':12,'d':13}
        return
    
    def chooseAction(self,positions):
        while True:
            try:
                rock = self.inpdict[input(Fore.BLUE+"From Location: ")]
                blank =self.inpdict[input(Fore.BLUE+"To Location: ")]
            except:
                print(Fore.RED+"Wrong input. Try again..")
                print(Style.RESET_ALL)
                continue
            
            if np.isin(rock,positions[:,0]) and np.isin(blank,positions[:,1]):
                return (rock,blank)
            else:
                print(Fore.RED+"Wrong input. Try again..")
                print(Style.RESET_ALL)
                continue
        return

    def addState (self,state):
        pass
    def feedReward(self,reward):
        pass
    def reset(self):
        pass

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
    if (len(sys.argv) == 2):
        if (sys.argv[1] == 'train'):
            p1 = Player("x")
            p2 = Player("o")

            st = State(14,p1,p2)
            st.play(100000)

            p1.savePolicy()
            p2.savePolicy()
        elif (sys.argv[1] == 'play-manual'):
            toss = np.random.randint(2)
            if (toss):
                print("You won the toss .. You play 'x' and begin the game")
                p2 = Player ("computer", exp_rate =0)
                p2.loadPolicy("policy_o")
                p1 = HumanPlayer("human")
            else:
                print("Computer won the toss .. You play 'o'")
                p1 = Player ("computer", exp_rate =0)
                p1.loadPolicy("policy_x")
                p2 = HumanPlayer("human")

            st = State(14,p1,p2)
            st.play2()
        elif sys.argv[1] == 'play-random':
            cwins = 0
            rwins = 0
            no_games = int(input("Enter the number of random games to play with trained model[1-10000]: "))
            if ((no_games >0) and (no_games <10001)):
                for g in range(no_games):
                    toss = np.random.randint(2)
                    if (toss):
                        # Randguy plays x
                        p2 = Player ("computer", exp_rate =0)
                        p2.loadPolicy("policy_o")
                        p1 = RandomPlayer("random")
                    else:
                        p1 = Player ("computer", exp_rate =0)
                        p1.loadPolicy("policy_x")
                        p2 = RandomPlayer("random")
                    st = State(14,p1,p2)
                    wonby = st.play2()
                    if(wonby == "computer"):
                        cwins += 1
                    else:
                        rwins += 1
                print(f'Played {no_games} games: computer won {cwins} at {((cwins/no_games)*100):.1f} %')


        elif sys.argv[1] == 'test-board':
            while(True):
                bstr = input("Input the board(type <end> to end): ")
                if (bstr == 'end'):
                    break
                os = input("Input offside : ")
                print(f'Cleared Board: {validate(bstr,os)}')

        else:
            print('Usage : python rlmain.py [train | play-manual | play-random|test-board]')
            print('\ttrain - do not use unless you are sure. It overwrites the existing model')
            print('\tplay-manual - Play the game with the trained model')
            print('\tplay-random - The model plays against random moves')
            print('\ttest-board - To pass the test cases.')
    else:
        print('Usage : python rlmain.py [train | play-manual | play-random|test-board]')
        print('\ttrain - do not use unless you are sure. It overwrites the existing model')
        print('\tplay-manual - Play the game with the trained model')
        print('\tplay-random - The model plays against random moves')
        print('\ttest-board - To pass the test cases.')
        
