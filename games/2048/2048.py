import pygame
import numpy as np
import random
import time
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../../")
from DQNAgent import DQNAgent

pygame.init()
BOARD_COLOUR = (255,255,255)
BACKGROUND = (0,0,0)
WIDTH = 500
HEIGHT = 500
SCOREBOARD_HEIGHT = 60
FONT = pygame.font.SysFont("Courier New", 38)
BOARD_SIZE = 4
SCORE = 0
gameOver = False


def drawBoard(screen, board):
    screen.fill(BACKGROUND)
    pygame.draw.lines(screen, BOARD_COLOUR, (0, 0), [(HEIGHT,0), (HEIGHT, WIDTH), (0, WIDTH), (0,0)], 5)
    pygame.draw.line(screen, BOARD_COLOUR, (0, WIDTH/4), (HEIGHT, WIDTH/4), 5)
    pygame.draw.line(screen, BOARD_COLOUR, (0, WIDTH/2), (HEIGHT, WIDTH/2), 5)
    pygame.draw.line(screen, BOARD_COLOUR, (0, WIDTH*3/4), (HEIGHT, WIDTH*3/4), 5)
    pygame.draw.line(screen, BOARD_COLOUR, (HEIGHT/4, 0), (HEIGHT/4, WIDTH), 5)
    pygame.draw.line(screen, BOARD_COLOUR, (HEIGHT/2, 0), (HEIGHT/2, WIDTH), 5)
    pygame.draw.line(screen, BOARD_COLOUR, (HEIGHT*3/4, 0), (HEIGHT*3/4, WIDTH), 5)
    drawScoreboard(screen)
    for (rowIndex, row) in enumerate(board):
        for (columnIndex, num) in enumerate(row):
            if num != 0:
                drawTile(screen, rowIndex, columnIndex, num)
    drawGameOver(screen)

def drawGameOver(screen):
    global SCORE
    global gameOver
    if gameOver:
        pygame.draw.rect(screen, BACKGROUND, (WIDTH/2 - 135, HEIGHT/2 - 40, 272, 82))
        string = "GAME OVER"
        font = pygame.font.SysFont("Courier New", 50)
        text = font.render(string, True, (0,0,255))
        screen.blit(text, (WIDTH/2 - 135, HEIGHT/2 - 40))
        string = "SCORE: " + str(int(SCORE))
        font = pygame.font.SysFont("Courier New", 40)
        text = font.render(string, True, (0,0,255))
        screen.blit(text, (WIDTH/2 - 135, HEIGHT/2))

def drawScoreboard(screen):
    global SCORE
    string = "SCORE: " + str(int(SCORE))
    text = FONT.render(string, True, BOARD_COLOUR)
    screen.blit(text, (0+5,HEIGHT+5))

def drawTile(screen, x, y, num):
    text = FONT.render(str(int(num)), True, BOARD_COLOUR)
    screen.blit(text, (x*WIDTH/4+5,y*HEIGHT/4+5))

def shift(board):
    for x in range(BOARD_SIZE):
        for y in range(1, BOARD_SIZE):
            if board[x][y] != 0:
                moveTo = y
                for i in range(y-1,-1,-1):
                    if board[x][i] == 0:
                        moveTo = i
                if moveTo != y:
                    board[x][moveTo] = board[x][y]
                    board[x][y] = 0

def move(board):
    global SCORE
    boardCopy = np.copy(board)
    shift(board)
    # add together
    for x in range (BOARD_SIZE):
        y = 0
        while y < BOARD_SIZE-1:
            if board[x][y] != 0:
                if board[x][y] == board[x][y+1]:
                    board[x][y+1] = 0
                    board[x][y] *= 2
                    SCORE += board[x][y]
                    y += 1
            y += 1
    shift(board)
    if not np.array_equal(boardCopy, board):
        newRandomTile(board)
        '''
        print("movement")
    else:
        print("no movement")'''
    return isOver(board)

def newRandomTile(board):
    while True:
        x = np.random.random_integers(0,3)
        y = np.random.random_integers(0,3)
        if board[x][y] == 0:
            if np.random.rand() < 0.1:
                board[x][y] = 4
            else:
                board[x][y] = 2
            break

def isOver(board):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == 0:
                return False
            if x+1 < BOARD_SIZE and board[x+1][y] == board[x][y]:
                return False
            if y+1 < BOARD_SIZE and board[x][y+1] == board[x][y]:
                return False
    return True

def moveDown(board):
    board = np.flip(board, axis=1)
    gameOver = move(board)
    board = np.flip(board, axis=1)
    return gameOver

def moveLeft(board):
    board = np.transpose(board)
    gameOver = move(board)
    board = np.transpose(board)
    return gameOver

def moveRight(board):
    board = np.transpose(board)
    board = np.flip(board, axis=1)
    gameOver = move(board)
    board = np.flip(board, axis=1)
    board = np.transpose(board)
    return gameOver

def init_game():
    global gameOver
    global SCORE
    SCORE = 0
    board = np.zeros((4,4))
    newRandomTile(board)
    newRandomTile(board)
    gameOver = False
    return board

def play2048():
    global gameOver
    screen = pygame.display.set_mode([WIDTH,HEIGHT + SCOREBOARD_HEIGHT])
    board = init_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                board = init_game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP and not gameOver:
                gameOver = move(board)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN and not gameOver:
                gameOver = moveDown(board)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT and not gameOver:
                gameOver = moveLeft(board)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT and not gameOver:
                gameOver = moveRight(board)
        drawBoard(screen, board)
        pygame.display.update()

def trainDQN(agent, scores):
    global gameOver
    plt.axis([0,1000,0,10000])
    plt.ion()
    plt.show()
    board = init_game()
    for gameNum in range(1000):
        board = init_game()
        moves = 0
        while not gameOver:
            prev_score = SCORE
            prev_board = np.copy(board)
            action = agent.act(np.reshape(board, [1,16]))

            if action == 0:
                gameOver = move(board)
            elif action == 1:
                gameOver = moveDown(board)
            elif action == 2:
                gameOver = moveLeft(board)
            elif action == 3:
                gameOver = moveRight(board)

            agent.memorize(np.reshape(prev_board, (16,)), action, SCORE - prev_score, np.reshape(board, (16,)), gameOver)
            moves += 1
            if moves % 50 == 0 or gameOver:
                agent.train(gameOver)
        print("Game Number:", gameNum, "Score:", SCORE, "Epsilon:", agent.epsilon, "Moves:", moves)
        scores.append(SCORE)
        plt.plot(scores)
        plt.draw()
        plt.pause(0.0001)
        agent.decayEpsilon()
        if gameNum % 10 == 0:
            print("saving", len(agent.memory))
            agent.save("saved/save/agent.hz")
            gamefile = open("saved/save/games.pkl","wb")
            pickle.dump([agent.memory, scores], gamefile, protocol=pickle.HIGHEST_PROTOCOL)
            gamefile.close()
    # final save
    agent.save("saved/save/agent.hz")
    gamefile = open("saved/save/games.pkl","wb")
    pickle.dump([agent.memory, scores], gamefile, protocol=pickle.HIGHEST_PROTOCOL)
    gamefile.close()



if __name__ == "__main__":
    agent = DQNAgent((16,), 4, 0.99, 512, 0.6, 0.01)
    agent.load("saved/save/agent.hz")

    gamefile = open("saved/save/games.pkl","rb")
    [agent.memory, scores] = pickle.load(gamefile)
    gamefile.close()

    trainDQN(agent, scores)
    #play2048()