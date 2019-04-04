import scipy.sparse as sp
import numpy as np
import pandas as pd

class Preprocess(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainAmazon = self.load_matrix(path + ".train.rating")   #Rating value
        self.testAmazon = self.load_list(path + ".test.rating")
           
        self.trainAmazon1 = self.load_matrix1(path + ".train.reliability")   #Reliability score
        self.testAmazon1 = self.load_list1(path + ".test.reliability")
        
        self.trainAmazon2 = self.load_matrix2(path + ".train.view")   #View score
        self.testAmazon2 = self.load_list2(path + ".test.view")
        
        self.num_users, self.num_items = self.trainAmazon.shape
        
    def load_list(self, filename):               
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList




   def load_list1(self, filename):
        reliabilityList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                reliabilityList.append([user, item])
                line = f.readline()
        return reliabilityList
    

   def load_list2(self, filename):
        viewList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                viewList.append([user, item])
                line = f.readline()
        return viewList
    
   ###################################################################### 
    
    
