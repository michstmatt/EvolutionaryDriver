import sys
import pygame
from pygame.locals import *
import time
import math
import random
import json
import numpy
import itertools

class Game:
    wayPoints = []
    wayPointIndex = 0
    mapData = []
    startPoint = []
    def currentWaypoint(self):
        return self.wayPoints[self.wayPointIndex]

class Car:
    xPos = 0
    yPos = 0
    angle = 0
    speed = 0
    acceleration = 5
    width = 20
    height = 20
    sensorDistance = 80
    speedMax = 15
    sensors = [-45,0,45]
    inputs = [0,0,0,0]
    outputs = [0,0,0,0]
    angleRate = 22.5

    def __init__(self,xPos,yPos,angle,speed):
        self.xPos = xPos
        self.yPos = yPos
        self.angle = angle
        self.speed = speed

    def copy(self):
        return Car(self.xPos,self.yPos,self.angle,self.speed)

    def move(self):

        self.speed *= 0.6
        
        if self.outputs[0] == 1:
            self.angle += self.angleRate
        elif self.outputs[1] == 1:
            self.angle -= self.angleRate
        
        if self.outputs[2] == 1 and self.speed >= self.acceleration:
            self.speed -= self.acceleration * 1.5
        elif self.outputs[3] == 1 and self.speed < self.speedMax:
            self.speed += self.acceleration
        #self.angle += max(min(self.outputs[0],1),-1) * self.angleRate
        #self.speed += max(min(self.outputs[1],1),-1) * self.acceleration
        
        if self.speed < 0:
            self.speed = 0
        if self.speed > self.speedMax:
            self.speed = self.speedMax
            
        moveX = math.cos(self.angle/57.3) * self.speed
        moveY = math.sin(self.angle/57.3) * self.speed
        
        self.xPos += moveX
        self.yPos -= moveY


class Driver:
    id = 0
    fitness = 0
    numHiddenLayers = 2
    hiddenLayerSize = 2
    # this is the weight vector for the "neural net" it has a weight for each input neuron to each output
    weights = []

    def __init__(self, weights):
        self.weights = weights

    '''
    This function generates a NN with random weight values between -1 and +1 for the inputs x outputs dimensional array
    '''
    def randInit():
        inputs = len(Car.inputs) + 1
        outputs = len(Car.outputs)

        layers = []
        # input to hidden
        layers.append(numpy.random.rand(inputs, Driver.hiddenLayerSize)*2 - 1)
        # hidden to hidden
        layers += [numpy.random.rand(Driver.hiddenLayerSize, Driver.hiddenLayerSize)
                   * 2 - 1 for i in range(Driver.numHiddenLayers-1)]
        # hidden to output
        layers.append(numpy.random.rand(Driver.hiddenLayerSize, outputs)*2 - 1)

        return Driver(layers)

    '''
    Applies the s curve to the function, ie low values => 0, higher => 1
    '''
    def sigmoid(x):
        return 1 / (1 + numpy.exp(-x))

    '''
    This function takes in an inputs vector which should be the same dimensionality as the inputWeight vector
    It generates a boolean vector for the specified number of outputs
    '''

    def think(self, inputs):

        dotProducts = numpy.array([1]+inputs.copy())

        for layer in self.weights:
            val = numpy.dot(dotProducts, layer)
            dotProducts = numpy.around(numpy.apply_along_axis(Driver.sigmoid, 0, val))

        # dot product, returns an array of dimension outputs

        # apply sigmoid to each output value and round
        outputs = dotProducts
        return outputs

    def thinkReg(self, inputs):
        dotProducts = inputs.copy()
        # dot product, returns an array of dimension outputs
        for layer in self.weights:
            val = numpy.dot(dotProducts, layer)
            dotProducts = [max(min(i, 1), -1) for i in val]

        return dotProducts


def createInitialPopulation(populationSize):
    generation = [Driver.randInit() for i in range(populationSize)]
    return generation


def loadWeights(populationFile):
    generation = []
    fileContent = open(populationFile).read()
    jsonData = json.loads(fileContent)

    for genome in jsonData:
        weights = [numpy.array(w) for w in genome]
        generation.append(Driver(weights))
    return generation


def storeWeights(elites, populationFile):
    formatted = []
    for gen in elites:
        formatted.append([i.tolist() for i in gen.weights])
    with open(populationFile, 'w') as file:
        json.dump(formatted, file)


def crossover(pair):
    weights = []

    split = int(len(pair[0].weights)/2)

    weights = pair[0].weights[:split] + list(pair[1].weights[split:])
    return Driver(weights)


def mutate(genome):
    genome.weights = [genome.weights[i] + (numpy.random.rand(genome.weights[i].shape[0], genome.weights[i].shape[1]) - 0.5) * .05 for i in range(len(genome.weights))]
    
    return genome


def evolve(generation, numElites):
    nextGeneration = []
    sortedGen = sorted(
        generation, key=lambda scored: scored.fitness, reverse=True)

    elites = sortedGen[:numElites]
    for gen in elites:
        print(gen.fitness, gen.weights)

    combinations = itertools.combinations(sortedGen[:numElites], 2)
    for pair in combinations:
        genome = crossover(pair)
        genome = mutate(genome)
        nextGeneration.append(genome)
        genome = crossover((pair[1], pair[0]))
        genome = mutate(genome)
        nextGeneration.append(genome)

    return (nextGeneration, elites)


def assessFitness(numWaypoints, time, distanceFromStart, totalDistance, speed):
    # return totalDistance
    return ((numWaypoints-1)**2) * 400 + totalDistance + distanceFromStart


def loadJsonCourse(fileName):
    game = Game()
    fileContent = open(fileName).read()
    jsonData = json.loads(fileContent)

    for line in jsonData["map_data"]:
        game.mapData.append([int(c) for c in line.strip()])

    game.wayPoints = jsonData["waypoints"]
    game.startPoint = jsonData["start"]

    return game


def checkCarIntersect(carRect, game):
    sXPos = int((carRect.left) / 20)
    sYPos = int((carRect.top) / 20)
    eXPos = int((carRect.right) / 20)
    eYPos = int((carRect.bottom) / 20)

    if sXPos < 0 or sYPos < 0 or eXPos >= len(game.mapData[0]) or eYPos >= len(game.mapData):
        return -1

    if game.mapData[sYPos][sXPos] + game.mapData[sYPos][eXPos] + game.mapData[eYPos][sXPos] + game.mapData[eYPos][eXPos] < 4:
        return 0

    if euclidDistance(carRect.center, game.currentWaypoint()) < 40:
        print("waypoint %d hit\n" % game.wayPointIndex)
        return 2

    return 1


def getCarSensors(carRect, car, game):
    pos = carRect.center
    sensorVals = []
    sensorPoints = []
    for angle in car.sensors:
        angle += car.angle

        sensorVal = int(car.sensorDistance/2)
        for sensorDistance in range(car.sensorDistance):

            sensorPoint = (pos[0] + math.cos(angle/57.3) * sensorDistance,
                           pos[1] - math.sin(angle/57.3) * sensorDistance)
            xIndex = int(sensorPoint[0]/20)
            yIndex = int(sensorPoint[1]/20)

            if xIndex >= len(game.mapData[0]) or yIndex >= len(game.mapData) or game.mapData[yIndex][xIndex] == 0:
                sensorVal = sensorDistance - sensorVal
                break

        sensorPoints.append(sensorPoint)
        sensorVals.append(2*sensorVal/car.sensorDistance)

    sensorVals.append(car.speed/car.speedMax)

    return (sensorVals, sensorPoints)


def drawCourse(game, assets, screen, sensorPoints):

    cellWidth = 20
    cellHeight = 20
    for rowIndex in range(len(game.mapData)):
        for cellIndex in range(len(game.mapData[rowIndex])):
            xPos = cellIndex * cellWidth
            yPos = rowIndex * cellHeight
            assetIndex = game.mapData[rowIndex][cellIndex]
            screen.blit(assets[assetIndex], (xPos, yPos))

    for wayPoint in game.wayPoints:
        screen.blit(assets[2], wayPoint)

    for sensorPoint in sensorPoints:
        screen.blit(assets[4], (sensorPoint[0],
                                sensorPoint[1] - assets[4].get_height()/2))


def euclidDistance(point1, point2):
    return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2) ** 0.5


def gameInit():
    pygame.init()
    grass = pygame.image.load("../resources/images/grass.png")
    road = pygame.image.load("../resources/images/road.png")
    player = pygame.image.load("../resources/images/circle.png")
    wayPoint = pygame.image.load("../resources/images/waypoint.png")
    course = pygame.image.load("../resources/images/course3.png")
    sensor = pygame.image.load("../resources/images/sensorPoint.png")
    assets = (grass, road, wayPoint, player, sensor)

    game = loadJsonCourse("../courses/6.json")

    dims = (1000, 1000)
    screen = pygame.display.set_mode(dims)
    return (game, screen, assets)


def gameMain(driver, game, screen, assets):

    player = assets[3]
    car = Car(game.startPoint[0], game.startPoint[1], 0, 0)
    lastCar = car.copy()

    game.wayPointIndex = 0

    stoppedTime = 0
    elapsedTime = 0
    distance = 0
    totalDistance = 0
    lastDistance = 0
    fitness = 0
    lastFitness = -1

    running = 1
    while running:
        pygame.display.set_caption('driving')
        screen.fill(0)

        playerRot = pygame.transform.rotate(player, car.angle)
        rect = playerRot.get_rect()
        rect.topleft = (car.xPos, car.yPos)

        (sensorVals, sensorPoints) = getCarSensors(rect, car, game)
        car.outputs = driver.think(sensorVals)
        car.move()

        state = checkCarIntersect(rect, game)

        if state == 2:
            game.wayPointIndex += 1
            game.wayPointIndex %= len(game.wayPoints)

        if state == 0:
            print("crash\r\n")
            running = 0

        if distance < 60 or car.speed < 0.5:
            stoppedTime += 1
        else:
            stoppedTime = 0

        if stoppedTime > 15 or elapsedTime > 1500:
            print("timeout\r\n",)
            running = 0

        drawCourse(game, assets, screen, sensorPoints)
        screen.blit(playerRot, (car.xPos, car.yPos))

        pygame.display.flip()
        time.sleep(0.02)
        elapsedTime += 1

        for event in pygame.event.get():
            # check if the event is the X button
            if event.type == pygame.QUIT:
                # if it is quit the game
                running = 0

        distance = euclidDistance((car.xPos, car.yPos), game.wayPoints[0])

        deltaDistance = euclidDistance(
            (car.xPos, car.yPos), (lastCar.xPos, lastCar.yPos))

        totalDistance += deltaDistance

        lastDistance = distance

        fitness = assessFitness(
            game.wayPointIndex, elapsedTime, distance, totalDistance, car.speed)
        lastFitness = fitness

        lastCar = car.copy()

        print("    Sensors: ", sensorVals, "L/R/Speed:",
              car.outputs, "Fitness:", fitness, end='\r')

    print('\r\n')
    return fitness


def main(argv):

    if len(argv) >= 2 and argv[0] == '-f':
        weightsFile = argv[1]
        generation = loadWeights(weightsFile)
    else:
        weightsFile = 'bestWeightsSig.json'
        generation = createInitialPopulation(100)

    (game, screen, assets) = gameInit()

    allGenomes = []

    for g in generation:
        print(g.weights)

    stop = input()

    for i in range(1000):

        for genome in generation:
            fitness = gameMain(genome, game, screen, assets)
            print(i, fitness)
            genome.fitness = fitness

        print("")
        print("")
        print("")
        print("--- NEW GENERATION --- ")
        print("")
        print("")
        print("")
        allGenomes += generation
        (generation, best) = evolve(allGenomes, 10)
        storeWeights(best, weightsFile)

    pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
