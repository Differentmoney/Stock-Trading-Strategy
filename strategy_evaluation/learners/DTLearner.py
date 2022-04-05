""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
      
import numpy as np          	  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
class DTLearner(object):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  	  			  		 			     			  	 
    :type verbose: bool  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    def __init__(self, leaf_size=1, verbose=False):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Constructor method  		  	   		  	  			  		 			     			  	 
        """  
        self.leaf_size=leaf_size	
        self.verbose=verbose
        self.tree=None	  	   		  	  			  		 			     			  	 
        pass  # move along, these aren't the drones you're looking for  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    def author(self):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        :return: The GT username of the student  		  	   		  	  			  		 			     			  	 
        :rtype: str  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        return "axiao31"  # replace tb34 with your Georgia Tech username  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 

    def buildTree(self, x, y):
        # base cases
        # cant split the tree, size data too small
        if x.shape[0]<=self.leaf_size:
            return np.array([[-1,y.mean(),-1,-1]])
        # cant split the tree, all data same value
        if len(np.unique(y))==1:
            return np.array([[-1,y[0],-1,-1]])

        # find the highest correlation value to do split
        corr=np.zeros(x.shape[1]-1)
        for i in range(x.shape[1]-1):
            corr[i]=abs(np.corrcoef(x[:,i],y)[0][1])
        index = np.argmax(corr)
        feature = x[:,index]
        split = np.median(feature)

        # check if feature is max, end if true
        if split==np.max(feature):
            return np.array([[-1, np.mean(y),-1,-1]])

        # split the feature
        l = feature <= split
        r = feature > split

        left = self.buildTree(x[l], y[l])
        right = self.buildTree(x[r], y[r])
        root = np.array([[index, split, 1, left.shape[0] + 1]])
        # print("root", root.shape)
        # print("left", left.shape)
        # print("right", right.shape)
        return np.vstack((root, left, right))
        
    def add_evidence(self, data_x, data_y):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 		 	     			  	 
        Add training data to learner  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param data_x: A set of feature values used to train the learner  		  	   		  	  			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		  	  			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 	   		  	  			  		 			     			  	 		     			  	 	  	  			  		 			     			  	 
        # build and save the model  		  	   		  	  			  		 			     			  	 
        self.tree= self.buildTree(data_x, data_y)	

    def query(self, points):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Estimate a set of test points given the model we built.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  	  			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		  	  			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        result = [0]
        for x in points:
            i = 0
            node =0
            while node != -1:
                node = int(self.tree[i, 0])
                if node == -1:
                    result.append(self.tree[i, 1])
                else:
                    if x[node] > self.tree[i, 1]:
                        i += int(self.tree[i, -1])
                    else:
                        i += int(self.tree[i, 2])

        return result[1:]
        


  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("the secret clue is 'zzyzx'")