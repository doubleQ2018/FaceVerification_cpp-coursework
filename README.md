# Boosted multi-task learning for face verification 
This is an implementation of [Boosted multi-task learning for face verification with applications to web image and video search](http://ieeexplore.ieee.org/document/5206736/?reload=true) in C++. The repository implemented Boosted Multi-Task Learning algorithm for face cerification task and did some experiments for the evaluation.

Includes:
* LBP(Local Binary Pattern)  Features Extraction
* Adaboost Learning with LBP features
* Boosted Multi-Task Learning

## LBP Features Extraction
* LBP Feature (Radius = 2, Points = 8)
<img src="assert/example.png" width="70%" />
* LBP results of different P&R values in my experiment
<img src="assert/LBP.png" width="59%" />

## Adaboost Learning with LBP Features
* ROC curve of different Iteration times and precision results
<img src="assert/iteration.png" width="59%" />
* Roc curve of different bins features
<img src="assert/bins.png" width="59%" />

## Boosted Multi-Task Learning
* Different Iteration times
<img src="assert/multi_iteration.png" width="59%" />
* ROC curves of verifying images of 40 celebrities
<img src="assert/multi_K.png" width="59%" />
