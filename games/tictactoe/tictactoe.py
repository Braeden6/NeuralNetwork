import pygame, time
import numpy as np
import tensorflow
import time
import sys
import pickle
import random
sys.path.append("..//..//")
from DQNAgent import DQNAgent


WHITE = (255,255,255)
SIZE = 150
DIMENSION = 3


class TICTACTOE:
    def __init__(self, display):
        self.board = [[0 for i in range(DIMENSION)] for j in range(DIMENSION)]
        self.player = 1 # 1 is for X, 2 is for O
        self.won = 0 # 1 X wins, 2 O wins, -1 Tie
        if display:
            self.screen = pygame.display.set_mode([SIZE*3, SIZE*3])

    def drawBoard(self):
        # Draw board
        for i in range(DIMENSION):
            for j in range(DIMENSION):
                pygame.draw.rect(self.screen, WHITE, (i*SIZE,j*SIZE,SIZE,SIZE),1)
                if self.board[i][j] == 1:
                    pygame.draw.line(self.screen, WHITE, (i*SIZE+10,j*SIZE+10), ((i+1)*SIZE-10,(j+1)*SIZE-10), 5)
                    pygame.draw.line(self.screen, WHITE, ((i)*SIZE+10,(j+1)*SIZE-10), ((i+1)*SIZE-10,j*SIZE+10), 5)
                if self.board[i][j] == 2:
                    pygame.draw.circle(self.screen, WHITE, (i*SIZE+SIZE/2, j*SIZE+SIZE/2), SIZE/2-10, 5)

    def areEqual(self,p1,p2,p3):
        return p1 == p2 == p3 and p1 != 0
                
    def detectGameDone(self, board):
        # down
        if self.areEqual(board[0][0], board[0][1], board[0][2]):
            return  board[0][0]
        if self.areEqual(board[1][0], board[1][1], board[1][2]):
            return   board[1][0]
        if self.areEqual(board[2][0], board[2][1], board[2][2]):
            return   board[2][0]
        # across
        if self.areEqual(board[0][0], board[1][0], board[2][0]):
            return   board[0][0]
        if self.areEqual(board[0][1], board[1][1], board[2][1]):
            return   board[0][1]
        if self.areEqual(board[0][2], board[1][2], board[2][2]):
            return   board[0][2]
        # dia
        if self.areEqual(board[0][0], board[1][1], board[2][2]):
            return   board[0][0]
        if self.areEqual(board[2][0], board[1][1], board[0][2]):
            return   board[2][0]
        # tie
        if np.count_nonzero(board) == 9:
            return -1
        return   0

    def getBoard(self):
        return self.board

    def setBoard(self, board):
        self.board = board
   
    def swapPlayers(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def playGame(self):
        pygame.init()
        # Run until the user asks to quit
        running = True
        while running:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if self.won != 0 and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print(self.won)
                    self.__init__(True)
                if event.type == pygame.MOUSEBUTTONDOWN and self.won == 0:
                    (x,y) = pygame.mouse.get_pos()
                    x = int(x/SIZE)
                    y = int(y/SIZE)
                    if self.board[x][y] == 0:
                        self.board[x][y] = self.player
                        self.swapPlayers()
                        self.won = self.detectGameDone(self.board) 
                self.drawBoard()
            # Flip the display
            pygame.display.flip()
        # Done! Time to quit.
        pygame.quit()
    
    def playagainstNN(self):
        playersTurn = False
        agent = DQNAgent((9,), 9, 0.99, 200, 0, 0)
        agent.load("saved/save5/agent1.hz")
        pygame.init()
        # Run until the user asks to quit
        running = True
        while running:
            if not playersTurn and self.won == 0:
                agentBoard = np.resize(self.board,(9,))
                action = getPossibleMove(agent, 0, agentBoard)
                agentBoard[action] = 2
                self.board = np.resize(agentBoard,(3,3))
                self.won = self.detectGameDone(self.board) 
                playersTurn = True
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if self.won != 0 and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    playersTurn = False
                    self.__init__(True)
                if event.type == pygame.MOUSEBUTTONDOWN and self.won == 0:
                    (x,y) = pygame.mouse.get_pos()
                    x = int(x/SIZE)
                    y = int(y/SIZE)
                    if self.board[x][y] == 0 and playersTurn:
                        self.board[x][y] = 1
                        self.won = self.detectGameDone(self.board) 
                        playersTurn = False
            self.drawBoard()
            # Flip the display
            pygame.display.flip()
        # Done! Time to quit.
        pygame.quit()

def getPossibleMove(agent, agentNum, board):
    action = agent.act_percentages(np.reshape(board, [1,9]))
    return np.argmax([a if 0 == b else 0 for b,a in zip(board, action)])

def getPossibleRandomMove(board):
    while(True):
        action = random.randrange(len(board))
        if board[action] == 0:
            return action

def trainTwoNN(agent1, agent2):
    game = TICTACTOE(False)
    TIES = 0
    AGENT1WINS = 0
    AGENT2WINS = 0
    NUM_OF_GAMES = 25
    
    for gameNum in range(1,20000):
        # play a game
        board = [0 for _ in range(9)]
        # reset for new board
        action1 = -1
        action2 = -1
        done = False

        if gameNum % NUM_OF_GAMES == 0:
            # agent1.updateTarget()
            # agent2.updateTarget()
            # Novice: 1 wins 57.1% 2 wins 30.6% and Ties 12.3%
            # Intermidiate: 1 wins 90.4% 2 wins 1.6% and Ties 8%
            # Expert: 1 wins 90.8% 2 wins 0.7% and Ties 8.5%
            print('Total games: {} Epsilon: {:.2f} TIES: {:.2f} Agent1 wins: {:.2f} Agent 2 wins: {:.2f}'.format(gameNum, agent1.epsilon, TIES/gameNum, AGENT1WINS/gameNum, AGENT2WINS/gameNum))

        while(not done):
            # agent 1 takes an action
            action1 = getPossibleMove(agent1, 1, board)

            # update game board
            beforeAgent1Board = board
            board[action1] = 1

            # check for win
            gameResult = game.detectGameDone(np.resize(board,(3,3)))
            #print(board, gameResult)

            # end game if over
            if gameResult != 0:
                if gameResult == -1:
                    # -5 reward if a tie
                    agent1.memorize(beforeAgent1Board, action1, 0, board, gameResult != 0)
                    agent2.memorize(beforeAgent2Board, action2, 0, board, gameResult != 0)
                    TIES += 1
                elif gameResult == 1:
                    # -10 reward for agent2 if agent1 wins
                    agent1.memorize(beforeAgent1Board, action1, 10, board, gameResult != 0)
                    agent2.memorize(beforeAgent2Board, action2, -10, board, gameResult != 0)#-10, board, gameResult != 0)
                    AGENT1WINS += 1
                done = True
                break
            else:
                # agent1 goes first so the first time we do not save
                if action2 != -1:
                    agent2.memorize(beforeAgent2Board, action2, 0, board, gameResult != 0)

            # agent 2 takes an action
            action2 = getPossibleMove(agent2, 2, board)

            # update game board
            beforeAgent2Board = board
            board[action2] = 2

            # check for win
            gameResult = game.detectGameDone(np.resize(board,(3,3)))
            # print(board, gameResult)

            # end game if over
            if gameResult != 0:
                if gameResult == -1:
                    # -5 reward if a tie
                    agent1.memorize(beforeAgent1Board, action1, 0, board, gameResult != 0)
                    agent2.memorize(beforeAgent2Board, action2, 0, board, gameResult != 0)
                    TIES += 1
                elif gameResult == 2:
                    # -10 reward for agent1 if agent2 wins
                    agent1.memorize(beforeAgent1Board, action1, -10, board, gameResult != 0)
                    agent2.memorize(beforeAgent2Board, action2, 10, board, gameResult != 0)
                    AGENT2WINS += 1
                done = True
                break
            else:
                # agent1 goes first so we always save
                agent1.memorize(beforeAgent1Board, action1, 0, board, gameResult != 0)


        agent1.train(0)
        agent2.train(0)
        agent1.decayEpsilon()
        agent2.decayEpsilon()
        if gameNum % 1000 == 0:
            agent1.save("saved/save5/agent1.hz")
            agent2.save("saved/save5/agent2.hz")
            file1 = open("saved/save5/games1.pkl","wb")
            file2 = open("saved/save5/games2.pkl","wb")
            pickle.dump(agent1.memory, file1)
            pickle.dump(agent2.memory, file1)
            file1.close()
            file2.close()

    agent1.save("saved/save5/agent1.hz")
    agent2.save("saved/save5/agent2.hz")


def trainAgaisntRandom(agent):
    game = TICTACTOE(False)
    TIES = 0
    AGENT1WINS = 0
    AGENT2WINS = 0
    NUM_OF_GAMES = 25
    
    for gameNum in range(1,20000):
        # play a game
        board = [0 for _ in range(9)]
        # reset for new board
        action1 = -1
        done = False

        if gameNum % NUM_OF_GAMES == 0:
            # Novice: 1 wins 57.1% 2 wins 30.6% and Ties 12.3%
            # Intermidiate: 1 wins 90.4% 2 wins 1.6% and Ties 8%
            # Expert: 1 wins 90.8% 2 wins 0.7% and Ties 8.5%
            print('Total games: {} Epsilon: {:.2f} TIES: {:.2f} Agent1 wins: {:.2f} Agent 2 wins: {:.2f}'.format(gameNum, agent.epsilon, TIES/gameNum, AGENT1WINS/gameNum, AGENT2WINS/gameNum))

        while(not done):
            # agent 1 takes an action
            action1 = getPossibleMove(agent, 1, board)

            # update game board
            beforeAgent1Board = board
            board[action1] = 1

            # check for win
            gameResult = game.detectGameDone(np.resize(board,(3,3)))
            #print(board, gameResult)

            # end game if over
            if gameResult != 0:
                if gameResult == -1:
                    # -5 reward if a tie
                    agent.memorize(beforeAgent1Board, action1, 0, board, gameResult != 0)
                    TIES += 1
                elif gameResult == 1:
                    # -10 reward for agent2 if agent1 wins
                    agent.memorize(beforeAgent1Board, action1, 10, board, gameResult != 0)
                    AGENT1WINS += 1
                done = True
                break

            # agent 2 takes an action
            board[getPossibleRandomMove(board)] = 2

            # check for win
            gameResult = game.detectGameDone(np.resize(board,(3,3)))
            # print(board, gameResult)

            # end game if over
            if gameResult != 0:
                if gameResult == -1:
                    # -5 reward if a tie
                    agent.memorize(beforeAgent1Board, action1, 0, board, gameResult != 0)
                    TIES += 1
                elif gameResult == 2:
                    # -10 reward for agent1 if agent2 wins
                    agent.memorize(beforeAgent1Board, action1, -10, board, gameResult != 0)
                    AGENT2WINS += 1
                done = True
                break
            else:
                # agent1 goes first so we always save
                agent.memorize(beforeAgent1Board, action1, 0, board, gameResult != 0)


        agent.train(0)
        agent.decayEpsilon()
        if gameNum % 1000 == 0:
            agent.save("saved/save6/agent.hz")
            file1 = open("saved/save6/games.pkl","wb")
            pickle.dump(agent.memory, file1)
            file1.close()

    agent.save("saved/save6/agent.hz")


def runGame():
    game = TICTACTOE(True)
    game.playagainstNN()

if __name__ == "__main__":
    agent = DQNAgent((9,), 9, 0.999, 128, 1, 0.01)
    #agent2 = DQNAgent((9,), 9, 0.999, 128, 1, 0.01)
    #agent1.load("saved/save3/agent1.hz")
    #agent2.load("saved/save3/agent2.hz")

    #trainTwoNN(agent1, agent2)
    trainAgaisntRandom(agent)

    #runGame()

