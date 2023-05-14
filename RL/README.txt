-------------------------------------------------------------------------------
Reinforcement Learning implementation of Alak
(Adapted from tic-tac-toe implementation of MJeremy
https://github.com/MJeremy2017/reinforcement-learning-implementation/blob/master/TicTacToe/ticTacToe.py)
-------------------------------------------------------------------------------

About the program
The program implements one dimensional variation of the game 'go' called 'Alak'. The implementation uses simple RL training method based on Bellman Equation. The program uses only the python library numpy for calculations.

How to use

The program supports 4 modes as given below

$ python rlmain.py train
1. Train : To train the RL model and generate policies for 'x' and 'o'. Do not use it unless you know what you are doing. Because it will override the already existing models.

$ python rlmain.py play-manual
2. play-manual : To play the game with the model. Program flips the coin and who goes first.

$ python rlmain.py play-random
3. play-random : The model plays against a random moves and finally prints the number of times the model won against the random moves. Presently, it is at 96.5%.

$ python rlmain.py test-board
4. test-board : Interactive way to test if program complies with the board clearing instructions.

