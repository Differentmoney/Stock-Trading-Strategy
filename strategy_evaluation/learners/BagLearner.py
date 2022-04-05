
import numpy as np
from scipy import stats

class BagLearner(object):  
    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):  
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learner_bag = []
        pass # move along, these aren't the drones you're looking for  

    def author(self):  
        return 'axiao31' # replace tb34 with your Georgia Tech user id.

    def add_evidence(self, data_x, data_y):

        for i in range(0, self.bags):
            #create random sample 
            index = np.random.choice(data_x.shape[0], np.shape(data_y)[0], replace=True)
            x = data_x[index]
            y = data_y[index]

            #create learner
            learner = self.learner(**self.kwargs)
            learner.add_evidence(x, y)

            #add learner to list
            self.learner_bag.append(learner)



    def query(self, points):
        r = np.empty([self.bags, points.shape[0]])
        for i in range(0, self.bags):
            r[i] = self.learner_bag[i].query(points)
        return stats.mode(r, axis=0)[0]
