import numpy
import sys
import scipy.sparse as sp

import pandas as pd


def op_strt(args):
    print args
    d= args[1]
    model = RatingRecsys()
    model.fit(d)


class RatingRecsys(object):
  

    def __init__(self):
        self.W = None
        self.Z = None

    def fit(self, user_x_product, latent_features_guess=5, learning_rate=0.01, steps=1000, regularization_penalty=0.02, convergeance_threshold=0.001):
        
        print 'training model...'
        return self.__factor_matrix(user_x_product, latent_features_guess, learning_rate, steps, regularization_penalty, convergeance_threshold)

    def predict_instance(self, row_index):
        
        return numpy.dot(self.Z[row_index, :], self.W.T)

    def predict_all(self):
        
        return numpy.dot(self.Z, self.W.T)

    def get_models(self):
        
        return self.Z, self.W

    def __factor_matrix(self, R, K, alpha, steps, beta, error_limit):
        
        
        R = numpy.array(R)

        
        N = len(R)
        M = len(R[0])
        Z = numpy.random.rand(N, K)

       
        W = numpy.random.rand(M, K)
        W = W.T

        error = 0

        
        for step in xrange(steps):

            
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:

                       
                        eij = R[i][j] - numpy.dot(Z[i, :], W[:, j])

                        for k in xrange(K):
                            
                            Z[i][k] = Z[i][k] + alpha * (2 * eij * W[k][j] - beta * Z[i][k])

                            
                            W[k][j] = W[k][j] + alpha * ( 2 * eij * Z[i][k] - beta * W[k][j] )


########################################
def op_strt(args):
    print args
    d1= args[1]
    model = ReliabilityRecsys()
    model.fit(d1)
class ReliabilityRecsys(object):
  

    def __init__(self):
        self.E = None
        self.F = None

    def fit(self, user_x_product, latent_features_guess=2, learning_rate=0.01, steps=1000, regularization_penalty=0.02, convergeance_threshold=0.001):
        
        print 'training model...'
        return self.__factor_matrix(user_x_product, latent_features_guess, learning_rate, steps, regularization_penalty, convergeance_threshold)

    def predict_instance(self, row_index):
        
        return numpy.dot(self.F[row_index, :], self.E.T)

    def predict_all(self):
        
        return numpy.dot(self.F, self.E.T)

    def get_models(self):
        
        return self.F, self.E

    def __factor_matrix(self, Re, K, alpha, steps, beta, error_limit):
        
        
        Re = numpy.array(Re)

        
        N = len(Re)
        M = len(Re[0])
        F = numpy.random.rand(N, K)

       
        E = numpy.random.rand(M, K)
        E = E.T

        error1 = 0

        
        for step in xrange(steps):

            
            for i in xrange(len(Re)):
                for j in xrange(len(Re[i])):
                    if Re[i][j] > 0:

                       
                        eij = Re[i][j] - numpy.dot(F[i, :], E[:, j])

                        for k in xrange(K):
                            
                            F[i][k] = F[i][k] + alpha * (2 * eij * E[k][j] - beta * F[i][k])

                            
                            E[k][j] = E[k][j] + alpha * ( 2 * eij * F[i][k] - beta * E[k][j] )

            
            

###############################################
def op_strt(args):
    print args
    d2= args[1]
    model = ViewRecsys()
    model.fit(d2)

class ViewRecsys(object):
  

    def __init__(self):
        self.W = None
        self.Z = None

    def fit(self, user_x_product, latent_features_guess=2, learning_rate=0.01, steps=1000, regularization_penalty=0.02, convergeance_threshold=0.001):
        
        print 'training model...'
        return self.__factor_matrix(user_x_product, latent_features_guess, learning_rate, steps, regularization_penalty, convergeance_threshold)

    def predict_instance(self, row_index):
        
        return numpy.dot(self.O[row_index, :], self.C.T)

    def predict_all(self):
        
        return numpy.dot(self.O, self.C.T)

    def get_models(self):
        
        return self.O, self.C

    def __factor_matrix(self, V, K, alpha, steps, beta, error_limit):
        
        
        V = numpy.array(V)

        
        N = len(V)
        M = len(V[0])
        O = numpy.random.rand(N, K)

       
        C = numpy.random.rand(M, K)
        C = C.T

        error = 0

        
        for step in xrange(steps):

            
            for i in xrange(len(V)):
                for j in xrange(len(V[i])):
                    if V[i][j] > 0:

                       
                        eij = V[i][j] - numpy.dot(O[i, :], C[:, j])

                        for k in xrange(K):
                            
                            O[i][k] = O[i][k] + alpha * (2 * eij * C[k][j] - beta * O[i][k])

                            
                            C[k][j] = C[k][j] + alpha * ( 2 * eij * O[i][k] - beta * C[k][j] )
########################################
       error = self.__error(R, Z, W, K, beta)

            
            if error < error_limit:
                break

        
        self.W = W.T
        self.Z = Z

        self.__print_fit_stats(error, N, M)

    def __error(self, R, Z, W, K, beta):
        
        e1 = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:

                   
                    e = e + pow(R[i][j]-numpy.dot(Z[i,:],W[:,j]), 2)

                    
                    for k in xrange(K):

                        
                        e1 = e1 + (beta/2) * ( pow(Z[i][k], 2) + pow(W[k][j], 2) )
        return e1
######################################################

error1 = self.__error1(Re, F, E, K, beta)

            
            if error1 < error_limit:
                break

        
        self.E = E.T
        self.F = F

        self.__print_fit_stats(error, N, M)

    def __error1(self, Re, F, E, K, beta):
        
        e1 = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if Re[i][j] > 0:

                    
                    e2 = e2 + pow(Re[i][j]-numpy.dot(F[i,:],E[:,j]), 2)

                  
                    for k in xrange(K):

                        
                        e2 = e2 + (beta/2) * ( pow(F[i][k], 2) + pow(E[k][j], 2) )
        return e2
##########################################################



error2 = self.__error2(V, O, C, K, beta)

            
            if error2 < error_limit:
                break

        
        self.C = C.T
        self.O = O

        self.__print_fit_stats(error, N, M)

    def __error(self, V, O, C, K, beta):
        
        e3 = 0
        for i in xrange(len(V)):
            for j in xrange(len(V[i])):
                if V[i][j] > 0:

                    
                    e3 = e3 + pow(V[i][j]-numpy.dot(O[i,:],C[:,j]), 2)

                    
                    for k in xrange(K):

                        
                        e3 = e3 + (beta/2) * ( pow(O[i][k], 2) + pow(C[k][j], 2) )
        return e3
  


    def __print_fit_stats(self, error, samples_count, products_count):
        print 'training complete...'
        print '------------------------------'
        print 'Stats:'
        print 'Error: %0.2f' % e_error
        print 'Samples: ' + str(samples_count)
        print 'Products: ' + str(products_count)
        print '------------------------------'

if __name__ == '__main__':
op_strt(sys.argv)
