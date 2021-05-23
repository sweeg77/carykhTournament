#This code will only work if you put it in the code folder for the tournament. 
#If you want the code to work outside the code folder you need to remove the classic strats part in RunGeneration(), runRound()
#Also, this code is pretty messy but works

import numpy as np
import random
import math
import importlib


def neuron(input,weights,bias):
    return 1/(1+math.e**(np.dot(weights,input)+bias))
pointsArray = [[1,5],[0,3]]



def strategy(history, memory):
    output = []
    if history.shape[1] == 0:        
        input = np.array([0,0,0,0,0,0])
    else:
        input = np.array([history.shape[1]/200,history[1][-1]])
        input = np.concatenate((input,memory[1]),axis=0)

    for i in range(4):
        output.append(neuron(input,memory[0][0][i],memory[0][1][i]))  
    return int(round(output[0])), [memory[0],output]
if __name__ == "__main__":
    class player():
        def __init__(self):
            self.weights = np.random.random((4,6))
            self.weights = (self.weights-0.5)*2
            self.score = 0
            self.bias = np.random.random(4)
            self.bias = (self.bias-0.5)*2
        #makes it into a copy of another Net with some mutations
        def GetCloned(self,weights,bias):
            mutation_degree = random.random()
            if  mutation_degree < 0.9:
                weightMutationMatrixPercent = np.random.rand(4,6)*0.2 +0.9
                weightMutationMatrixConstant = np.random.rand(4,6)*0.1 - 0.05

                biasMutationVectorPercent = np.random.rand(4)*0.2 +0.9
                biasMutationVectorConstant = np.random.rand(4)*0.1 - 0.05
            else:
                weightMutationMatrixPercent = np.random.rand(4,6)*0.6 +0.7
                weightMutationMatrixConstant = np.random.rand(4,6)*0.3 - 0.15

                biasMutationVectorPercent = np.random.rand(4)*0.6 +0.7
                biasMutationVectorConstant = np.random.rand(4)*0.3 - 0.15           

            self.weights = weights*weightMutationMatrixPercent + weightMutationMatrixConstant
            self.bias = bias*biasMutationVectorPercent + biasMutationVectorConstant

#Just a few functions from the tournament sourcecode so I can run tournaments here 
    def GetVisibleHistory(turn,History,player):
        historySoFar = History[:,:turn].copy()
        if player == 1:
            historySoFar = np.flip(historySoFar,0)
        return historySoFar

    def GetScores(history):
        scoreA = 0

        gameLength = history.shape[1]
        for turn in range(gameLength):
            scoreA += pointsArray[history[0,turn]][history[1,turn]]
        return scoreA/gameLength
    def strategyMove(move):
        if type(move) is str:
            defects = ["defect","tell truth"]
            return 0 if (move in defects) else 1
        else:
            return move
    def runRound(playerA,playerB):
        Round_length = int(200-40*np.log(random.random()))
        History = np.zeros((2,Round_length),dtype=int)
        classic = False
        MemoryA = [[playerA.weights,playerA.bias],[]]

        if type(playerB) is player:
            MemoryB = [[playerB.weights,playerB.bias],[]]
        else:
            MemoryB = None
            classic = True
            module = importlib.import_module(playerB)

        for turn in range(Round_length):
            playerAMove, MemoryA = strategy(GetVisibleHistory(turn,History,0),MemoryA)
            if classic:
                playerBMove, MemoryB = module.strategy(GetVisibleHistory(turn,History,1),MemoryB)  
                playerBMove = strategyMove(playerBMove)
            else:
                playerBMove, MemoryB = strategy(GetVisibleHistory(turn,History,1),MemoryB)  

            History[0,turn] = playerAMove
            History[1, turn] = playerBMove
        return GetScores(History)

    #NeuralNets must have an even number of networks
    #The networks only gets to compete agianst 2x Num_contenders
    def RunGeneration(NeuralNets,Num_contenders=15):
        classic_strats = ["exampleStrats.titForTat","exampleStrats.simpleton","exampleStrats.random","exampleStrats.joss","exampleStrats.grimTrigger","exampleStrats.ftft","exampleStrats.detective","exampleStrats.alwaysTellTruth","exampleStrats.alwaysStaySilent"]
        for player in NeuralNets:
            contenders = [[classic_strats[random.randint(0,8)] for i in range(Num_contenders)],[NeuralNets[random.randint(0,len(NeuralNets)-1)] for i in range(Num_contenders)]]
            Scores = []
            for classic in contenders[0]:
                Scores.append(runRound(player, classic))
            for Net in contenders[1]:
                Scores.append(runRound(player, Net))
            player.score = sum(Scores)/(Num_contenders*2)
        NeuralNets.sort(key=lambda x: x.score,reverse=True)
        netSize = int(len(NeuralNets)/2)
        for i in range(netSize):
            NeuralNets[i+netSize].GetCloned(NeuralNets[i].weights,NeuralNets[i].bias)
        return NeuralNets


    #runs simulation 
    Generaition = [player() for i in range(100)]
    for i in range(100):
        Generaition = RunGeneration(Generaition)
        print("Generation",i+1,"complete")
    print("best score: ", Generaition[0].score)
    print("Best weights:\n",Generaition[0].weights)
    print("Best bias:\n",Generaition[0].bias)


