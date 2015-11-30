# python_machine_learning

PCA(n_out=500) -> Random Forest(n_estimators = 100,'entropy')
n = 50000
n_train = 37500
n_test = 12500  
time: 126.9m
f-score: 0.44

Conducted walk-through on the experiment above by fitting/testing the model with moving windows in time. 
The f-score was stable (ranged from 0.43 to 0.44) throughout the walkthrough.

RBM(n_out=500) -> RandomForest(n_estimators = 100, 'entropy') 
n = 50000
n_train = 37500
n_test = 12500
f-score: 0.44
*Experimented with different values for n_out, but f-score didn't change significantly
*Also, experimented with multiple layers of RBM, but f-score still didn't change much
*Trained for longer time period (n_train = 87500) and tested on same set, but still got similar f-score.

RBM(n_out=1000) -> LogisticRegression
Since sklearn's LogisticRegression does not support labels with multiple columns, LogisticRegression was performed label by label for a total of 43 times. 
Also, used grid-search to get optimal value of 'C' (inverse of regularization strength).
n = 50000
n_train = 37500
n_test = 12500
f-score = 0.36













