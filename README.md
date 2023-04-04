# Kalman-filters
Two classes for standard Kalman filter and extended Kalman filter are implemented for two benchmark problems.  

## Dependency
- numpy
- matplotlib

## Benchmark
The standard Kalman filter is applied to predict the state variables from a linear Gaussian process system which is composed of a evoluation model and a measurement model. The corresponding problem statement can be summarized as follows:
![kalman filter](./problem_statement/problem1.PNG)  

With Kalman filtering, the predictions for position, velocity and acceleration variables can be given in the following figures:
|                    Position                    |                 Velocity                     |                      Acceleration                    |
:----------------------------------------------: | :--------------------------------------------: | :----------------------------------------------------: 
![position](./problem1_results/position.png)    | ![velocity](./problem1_results/velocity.png)  | ![acceleration](./problem1_results/acceleration.png)  


**To use standard Kalman filtering, run**
```
python example1.py
```
