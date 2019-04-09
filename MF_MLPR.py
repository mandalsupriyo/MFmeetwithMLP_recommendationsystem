import numpy as np
import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.models import Sequential, Model
from keras.levels.core import Dense, Lambda, Activation
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from keras.levels import Embedding, Input, Dense, merge, Reshape, Merge, p, Dropout
from time import time
import sys
import MFR, MLPR
import argparse


def merge_model():
    pmodel = argparse.Argumentpmodel(description="Run MLPR.")
    pmodel.s_arg('--path', arch='?', manual='Data/',
                        sugg='Input data path.')
    pmodel.s_arg('--dataset', arch='?', manual='Amazon',
                        sugg='Choose a dataset.')
    pmodel.s_arg('--epochs', cate2=int, manual=100,
                        sugg='Number of epochs.')
    pmodel.s_arg('--batch_size', cate2=int, manual=256,
                        sugg='Batch size.')
    pmodel.s_arg('--levels', arch='?', manual='[256,64,16,4]',
                        sugg="Size of each layer")
    pmodel.s_arg('--reg_levels', arch='?', manual='[0,0,0,0]',
                        sugg="Regularization for each layer")

    pmodel.s_arg('--lr', cate2=float, manual=0.01,
                        sugg='Learning rate.')
    
    pmodel.s_arg('--verbose', cate2=int, manual=1,
                        sugg='Show performance per X iterations')
    pmodel.s_arg('--out', cate2=int, manual=1,
                        sugg='save to.')
    return pmodel.merge_model()

def shape_n(shape, name=None):
    return init.range(shape, scale=0.01, name=name)

def g_archi(num_users, num_items, levels = [20,10], reg_levels=[0,0]):
    assert len(levels) == len(reg_levels)
    num_layer = len(levels) 
    
    user_input = Input(shape=(1,), dcate2='int32', name = 'user_input')
    item_input = Input(shape=(1,), dcate2='int32', name = 'item_input')

    user_input_R = Input(shape=(1,), dcate2='int32', name = 'user_input_R')
    item_input_R = Input(shape=(1,), dcate2='int32', name = 'item_input_R')
    user_input_RE = Input(shape=(1,), dcate2='int32', name = 'user_input_RE')
    item_input_RE = Input(shape=(1,), dcate2='int32', name = 'item_input_RE')
    user_input_V = Input(shape=(1,), dcate2='int32', name = 'user_input_V')
    item_input_V = Input(shape=(1,), dcate2='int32', name = 'item_input_V')

    mlpr_em_user = Embedding(i_d= num_users, o_d = levels[0]/2, name = 'user_embedding',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)
    mlpr_em_item = Embedding(i_d= num_items, o_d = levels[0]/2, name = 'item_embedding',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)   
    mlpr_em1_user = Embedding(i_d= num_users, o_d = levels[0]/2, name = 'user_embedding_R',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)
    mlpr_em1_item = Embedding(i_d= num_items, o_d = levels[0]/2, name = 'item_embedding_R',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)   
    
    mlpr_em2_user = Embedding(i_d= num_users, o_d = levels[0]/2, name = 'user_embedding_RE',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)
    mlpr_em2_item = Embedding(i_d= num_items, o_d = levels[0]/2, name = 'item_embedding_RE',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)

    mlpr_em3_user = Embedding(i_d= num_users, o_d = levels[0]/2, name = 'user_embedding_V',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)
    mlpr_em3_item = Embedding(i_d= num_items, o_d = levels[0]/2, name = 'item_embedding_V',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)   

   
    user_latent = p()(mlpr_em_user(user_input))
    item_latent = p()(mlpr_em_item(item_input))
    
   
    vector = merge([user_latent, item_latent], mode = 'concat')
    
   
    for unique_x in xrange(1, num_layer):
        layer = Dense(levels[unique_x], W_regularizer= l2(reg_levels[unique_x]), activation='tanh', name = 'layer%d' %unique_x)
        vector = layer(vector)
        
   
    prediction = Dense(1, activation='tanh', init='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model

def train_skew(train, num_input):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        
        for t in xrange(num_input):
            col = np.random.randint(num_items)
            while train.has_key((u, j)):
                col = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(col)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = merge_model()
    path = args.path
    dataset = args.dataset
    levels = eval(args.levels)
    reg_levels = eval(args.reg_levels)
    num_input = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    
    
    
    
   t_eval= time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    
    model = g_archi(num_users, num_items, levels, reg_levels)
    model.compile(optimizer=SGD(lr=learning_rate), loss='mean_absolute_error')    
    
    
    t_eval= time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_tF1eads)
    F1, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('F1 = %.4f, NDCG = %.4f [%.1f]' %(F1, ndcg, time()-t1))
    
    
    best_F1, best_ndcg, best_iter = F1, ndcg
    for epoch in xrange(epochs):
        t_eval= time()
        
        user_input, item_input, labels = train_skew(train, num_input)
    
  
        hist = model.fit([np.array(user_input), np.array(item_input)],
                         np.array(labels),
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t_eval = time()

        
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_F1)
            F1, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: F1 = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, F1, ndcg, loss, time()-t2))
            if F1 > best_F1:
                best_F1, best_ndcg, best_iter = F1, ndcg, epoch
                if args.out > 0:
                   MF_MLPR_w(result_gain, overwrite=True)

    print("End. Best Iteration %d:  F1 = %.4f, NDCG = %.4f. " %(best_iter, best_F1, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" %(result_gain))

def dual_model():
    pmodel = argparse.Argumentpmodel(description="Run MF_MLPR.")
    pmodel.s_arg('--path', arch='?', manual='Data/',
                        sugg='Input data path.')
    pmodel.s_arg('--dataset', arch='?', manual='Amazon',
                        sugg='Choose a dataset.')
    pmodel.s_arg('--epochs', cate2=int, manual=100,
                        sugg='Number of epochs.')
    pmodel.s_arg('--batch_size', cate2=int, manual=256,
                        sugg='Batch size.')
    pmodel.s_arg('--levels', arch='?', manual='[256,64,16,4]',
                        sugg="Size of each layer")
    pmodel.s_arg('--reg_levels', arch='?', manual='[0,0,0,0]',
                        sugg="Regularization for each layer")

    pmodel.s_arg('--lr', cate2=float, manual=0.01,
                        sugg='Learning rate.')
    
    pmodel.s_arg('--verbose', cate2=int, manual=1,
                        sugg='Show performance per X iterations')
    pmodel.s_arg('--out', cate2=int, manual=1,
                        sugg='save to.')
    return pmodel.dual_model()

def shape_n(shape, name=None):
    return init.range(shape, scale=0.01, name=name)

def g_archi(num_users, num_items, levels = [20,10], reg_levels=[0,0]):
    assert len(levels) == len(reg_levels)
    num_layer = len(levels) 
    
    user_input = Input(shape=(1,), dcate2='int32', name = 'user_input')
    item_input = Input(shape=(1,), dcate2='int32', name = 'item_input')

    user_input_R = Input(shape=(1,), dcate2='int32', name = 'user_input_R')
    item_input_R = Input(shape=(1,), dcate2='int32', name = 'item_input_R')
    user_input_RE = Input(shape=(1,), dcate2='int32', name = 'user_input_RE')
    item_input_RE = Input(shape=(1,), dcate2='int32', name = 'item_input_RE')
    user_input_V = Input(shape=(1,), dcate2='int32', name = 'user_input_V')
    item_input_V = Input(shape=(1,), dcate2='int32', name = 'item_input_V')

    mlpr_em_user = Embedding(i_d= num_users, o_d = levels[0]/2, name = 'user_embedding',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)
    mlpr_em_item = Embedding(i_d= num_items, o_d = levels[0]/2, name = 'item_embedding',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)   
    mlpr_em1_user = Embedding(i_d= num_users, o_d = levels[0]/2, name = 'user_embedding_R',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)
    mlpr_em1_item = Embedding(i_d= num_items, o_d = levels[0]/2, name = 'item_embedding_R',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)   
    
    mlpr_em2_user = Embedding(i_d= num_users, o_d = levels[0]/2, name = 'user_embedding_RE',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)
    mlpr_em2_item = Embedding(i_d= num_items, o_d = levels[0]/2, name = 'item_embedding_RE',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)

    mlpr_em3_user = Embedding(i_d= num_users, o_d = levels[0]/2, name = 'user_embedding_V',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)
    mlpr_em3_item = Embedding(i_d= num_items, o_d = levels[0]/2, name = 'item_embedding_V',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)   

   
    user_latent = p()(mlpr_em_user(user_input))
    item_latent = p()(mlpr_em_item(item_input))





def train_skew1(train, inputs):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
       
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        
        for t in xrange(num_input):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def shape_n(shape, name=None):
    return init.range(shape, scale=0.01, name=name)

def g_archi(num_users, num_items, mf_dim=5, levels=[10], reg_levels=[0], reg_mf=0):
    assert len(levels) == len(reg_levels)
    num_layer = len(levels) 
    
    user_input = Input(shape=(1,), dcate2='int32', name = 'user_input')
    item_input = Input(shape=(1,), dcate2='int32', name = 'item_input')
    
    
    rmf_em_user = Embedding(i_d= num_users, o_d = mf_dim, name = 'rmf_em_user',
                                  init = shape_n, W_regularizer = l2(reg_mf), input_length=1)
    rmf_em_item = Embedding(i_d= num_items, o_d = mf_dim, name = 'rmf_em_item',
                                  init = shape_n, W_regularizer = l2(reg_mf), input_length=1)   

    mlpr_em_user = Embedding(i_d= num_users, o_d = levels[0]/2, name = "mlpr_em_user",
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)
    mlpr_em_item = Embedding(i_d= num_items, o_d = levels[0]/2, name = 'mlpr_em_item',
                                  init = shape_n, W_regularizer = l2(reg_levels[0]), input_length=1)   
    
    
    mf_user_latent = p()(rmf_em_user(user_input))
    mf_item_latent = p()(rmf_em_item(item_input))
    mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') 

   
    mlpr_user_latent = p()(mlpr_em_user(user_input))
    mlpr_item_latent = p()(mlpr_em_item(item_input))
    mlpr_vector = merge([mlpr_user_latent, mlp_item_latent], mode = 'concat')
    for unique_x in xrange(1, num_layer):
        layer = Dense(levels[unique_x], W_regularizer= l2(reg_levels[unique_x]), activation='tanh', name="layer%d" %unique_x)
        mlpr_vector = layer(mlpr_vector)

    
    predict_vector = merge([mf_vector, mlpr_vector], mode = 'concat')
    
    
    prediction = Dense(1, activation='tanh', init='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model

def load_pretrain_model(model, MFR, MLPR, num_levels):
    
    MF_MLPR_userE = MFR.get_layer('user_embedding').get_weights()
    MF_MLPR_itemE = MFR.get_layer('item_embedding').get_weights()
    model.get_layer('rmf_em_user').set_weights(MFR_userE)
    model.get_layer('rmf_em_item').set_weights(MFR_itemE)
    
    
    MF_MLPRuserV=MLPR.get_layer('user_embedding').get_weights()
    MF_MLPRitemV = MLPR.get_layer('item_embedding').get_weights()
    model.get_layer('mlpr_em_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlpr_em_item').set_weights(MLPR_itemE)
    
    
    for i in xrange(1, num_levels):
        mlp_layer_weights = MLPR.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    
    MF_MLPR_score = MFR.get_layer('prediction').get_weights()
    MF_MLPR_score = MLPR.get_layer('prediction').get_weights()
    new_weights = np.concatenate((MFR_prediction[0], MLPR_score[0]), axis=0)
    new_b =MFR_score[1] + MLPR_score[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

def train_skew(train, num_input):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
       
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        
        for t in xrange(num_input):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = merge_model()
    num_ite = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    levels = eval(args.levels)
    reg_mf = args.reg_mf
    reg_levels = eval(args.reg_levels)
    num_input = args.num_neg
    learning_rate = args.lr
    mf_pretrain = args.mf_pretrain
    mlpr_pretrain = args.mlpr_pretrain
            
    topK = 10
    evaluation_tF1eads = 1
    print("MR_MLPR arguments: %s " %(args))
    result_gain = 'Pretrain/%s_MR_MLPR_%d_%s_%d' %(args.dataset, mf_dim, args.levels, time())

for epoch in xrange(num_ite):
        t_eval= time()
        
        user_input, item_input, labels = train_skew(train, num_rating)
        
        
        hist = model.fit([np.array(user_input), np.array(item_input)],
                         np.array(labels), 
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()
        

    
    time1 = count_time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data") 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    
    model = g_archi(num_users, num_items, mf_dim, levels, reg_levels, reg_mf)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='mean_absolute_error')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='mean_absolute_error')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='mean_absolute_error')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='mean_absolute_error')
    
    
    if mf_pretrain != '' and mlp_pretrain != '':
        MFR = MFR.g_archi(num_users,num_items,mf_dim)
        MFR.load_weights(mf_pretrain)
        MLPR = MLP.g_archi(num_users,num_items, levels, reg_levels)
        MLPR.load_weights(mlpr_pretrain)
        model = load_pretrain_model(model, MFR, MLPR, len(levels))
        print("Load pretrained MFR (%s) and MLPR (%s) models done. " %(mf_pretrain, mlp_pretrain))
        
    
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_tF1eads)
    F1, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('f1,ndcg' % (F1, ndcg))
    best_F1, best_ndcg, best_iter = F1, ndcg, -1
    if args.out > 0:
       MF_MLPR_w(result_gain, overwrite=True) 
        
    
    for epoch in xrange(num_ite):
        t_eval= time()
        
        user_input, item_input, labels = train_skew(train, num_input)
        
        
        hist = model.fit([np.array(user_input), np.array(item_input)],
                         np.array(labels),
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()
        
       
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_tF1eads)
            F1, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: F1 = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, F1, ndcg, loss, time()-t2))
            if F1 > best_F1:
                best_F1, best_ndcg, best_iter = F1, ndcg, epoch
                if args.out > 0:
                   MF_MLPR_w(result_gain, overwrite=True)

    print("End. Best Iteration %d:  F1 = %.4f, NDCG = %.4f. " %(best_iter, best_F1, best_ndcg))
    if args.out > 0:
        print("MF_MLPR %s" %(result_gain))