# Kalman-filters
Two classes for standard Kalman filter and extended Kalman filter are implemented for two benchmark problems in the file "kalman_filters.py".  

## Dependency
- numpy
- matplotlib

## Benchmark for standard Kalman filtering
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

## Benchmark for extended Kalman filtering
The extended Kalman filter is used to predict the state variables from a nonlinear Gaussian process system which is composed of a linear evoluation model and a nonlinear measurement model. The corresponding problem statement is given as follows:
![extended_kalman filter](./problem_statement/problem2.PNG)  

With extended Kalman filtering, the predictions for model parameters trajectories can be given in the following figures:
|                    *h*                    |                 $\theta$                     |  
:----------------------------------------------: | :--------------------------------------------:
![h](./problem2_results/h.png)                   | ![theta](./problem2_results/theta.png)  |
|                    $\omega$                     |                 $\alpha$                     |  
![omega](./problem2_results/omega.png)    | ![alpha](./problem2_results/alpha.png)  |

**To use extended Kalman filtering, run**
```
python example2.py
```
