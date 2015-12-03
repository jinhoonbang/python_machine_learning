# python_machine_learning

<h2> Experiment Results </h2>

<p>
PCA(n_out=500) -> Random Forest(n_estimators = 100,'entropy') <br/>
n = 50000 <br/>
n_train = 37500 <br/>
n_test = 12500 <br/>
time: 126.9m <br/>
f-score: 0.44 <br/>
classifcation error:0.46 <br/>
</p>

<p>
Conducted walk-through on the experiment above by fitting/testing the model with moving windows in time. 
The f-score was stable (ranged from 0.43 to 0.44) throughout the walkthrough.
</p>
<p>
RBM(n_out=500) -> RandomForest(n_estimators = 100, 'entropy')<br/>
n = 50000<br/>
n_train = 37500<br/>
n_test = 12500<br/>
f-score: 0.44<br/>
classification error:0.46<br/>
*Experimented with different values for n_out, but f-score didn't change significantly
*Also, experimented with multiple layers of RBM, but f-score still didn't change much
*Trained for longer time period (n_train = 87500) and tested on same set, but still got similar f-score.
</p>
<p>
RBM(n_out=1000) -> LogisticRegression<br/>
Since sklearn's LogisticRegression does not support labels with multiple columns, LogisticRegression was performed label by label for a total of 43 times. 
Also, used grid-search to get optimal value of 'C' (inverse of regularization strength).<br/>
n = 50000<br/>
n_train = 37500<br/>
n_test = 12500<br/>
f-score = 0.36<br/>
</p>












