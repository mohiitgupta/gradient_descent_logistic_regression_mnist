The submission contains report.pdf, this readme file, gd.py and sgd.py which contain the main logic for batch gradient descent and stochastic gradient descent respectively, util.py which contains util functions used for preprocessing and inference, graph_plotting.py which contains function for plotting the graph.

To run the code, kindly run as below:
 python gd.py True type2 ../DATA_FOLDER

 Here, DATA_FOLDER contains all the gz files and the above run corresponds to running batch gradient descent with regularization as True and type2 features.

 Kindly note, that for batch gradient descent, each epoch takes around 1.5 seconds on the data.cs.purdue.edu server. When I last checked the convergence happened after around 170 epochs with test accuracy of around 0.902 for type 1 features.

 For stochastic gradient descent, each epoch takes less than 1 second, the main reason being that here 1 epoch is defined as weight updates on 1000 examples. So, in total, the convergence happens maximum 300 epochs with test accuracy of around 0.907.

 We can observe that stochastic gradient descent has marginally higher test scores compared to batch gradient descent, the primary reason being that batch gradient descent requires much higher number of complete training data set passes to converge compared to stochastic gradient descent.