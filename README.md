# Evolutionary Driver

This project simulates a top down driving game, where the "driver" learns to 
drive via evolutionary computation (genetic algorithms)

## Inputs
The driver has "sensors" that reports the distance to the grass up to a set max
distance

## Outputs
The Driver can turn left, right, forward and stop


## Reg (Regression)
This model is takes input from sensors and produces a float value between -1 and 
+1 for all outputs 

## Sig (Sigmoid)
Same as regression but coherces the value to either -1 or +1 (no floating 
points) for all outputs

## Multilayer
There is also multilayer which is a neural network, one with neurons that do 
regression and the other does a sigmoid

## Courses
Courses are stored as JSON with map data 1 for road, 0 for grass

### Waypoints
These waypoints are used for the fitness function of the Genetic Algorithm (GA). 
The "drivers" have no idea about the waypoints are soley used for evalutaion.

## Notes
Lots of copy and pasted code. Only external dependencies are numpy for math and pygame for rendering
