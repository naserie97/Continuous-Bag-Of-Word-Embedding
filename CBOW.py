import numpy as np 
import nltk
import re
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from collections import defaultdict
import argparse
import pickle


# inputs
###########################################
# txt file  
# C: context half-size
# N: dimension of hidden vector 
# num_iters: number of iterations
# emb_weight_matrix: extract the word embedding from w1, w2 or combination of both
###########################################

# taking arguments from command line
##################################################################

ap = argparse.ArgumentParser()

ap.add_argument("-t", "--text_file", required=True,
   help="path to the text file, text file will contain text that consists of paragrapghs separated by \\n")

ap.add_argument("-c", "--half_size", required=True,
   help="context half-size")

ap.add_argument("-n", "--emb_dim", required=True,
   help="dimension of hidden vector ")

ap.add_argument("-e", "--num_iters", required=True,
   help="number of iteration for training the model")

ap.add_argument("-w", "--emb_weight_matrix", required=True,
   help="extract the word embedding from w1, w2 or combination of both, you can enter one of: w1, w2, w1w2")


args = vars(ap.parse_args())


# data preprocessing
##################################################################
# reading the text data

text_file = args['text_file']
with open (text_file) as file:
    data = file.read()


# replace the punctuation marks with a dot 
data = re.sub(r'[,!?;-]', '.',data)

# tokenize the dataset 
data = nltk.word_tokenize(data) 


# lowercase the tokens 
data = [token.lower() for token in data if token.isalpha() or token == '.']


# fdist is a frequency distribution of the words in the dataset
fdist = nltk.FreqDist(word for word in data)


def get_dict(data):
    """
    Input:
        data: the data you want to pull from
    Output:
        word2Ind: returns dictionary mapping the word to its index
        Ind2Word: returns dictionary mapping the index to its word
    """
    #
#     words = nltk.word_tokenize(data)
    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    # return these correctly
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word


# get_dict creates two dictionaries, converting words to indices and viceversa.
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)


# get batches of data for training 
##################################################################
def get_idx(words, word2Ind): 
    
    idx = []
    for word in words:
        idx = idx + [word2Ind[word]]
    return idx



def pack_idx_with_frequency(context_words, word2Ind):
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2Ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed


def get_vectors(data, word2Ind, V, C):
    i = C
    while True:
        # place holder in the size of vocab for the one hot encoding for each word in x
        y = np.zeros(V)
        # place holder in the size of vocab for the one hot encoding for each word in y
        x = np.zeros(V)
        # get the center word
        center_word = data[i]
        # one hot encoding for the word, put 1 in the zero vector in the index of the word
        y[word2Ind[center_word]] = 1
        # get the context words
        context_words = data[(i - C):i] + data[(i+1):(i+C+1)]
        num_ctx_words = len(context_words)
        # pack_idx_with_frequency return a list that contain tuples
        # each tupe is a word indext that was in the context word
        # and the frequency of this word in the given context
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            x[idx] = freq/num_ctx_words
        yield x, y
        i += 1
        if i >= len(data):
            print('i is being set to 0')
            i = 0

            
def get_batches(data, word2Ind, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2Ind, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch = []


# Training process
##################################################################
#initialize the weights and biases
def initialize_model(N,V, random_seed=1):
    '''
    Inputs: 
        N:  dimension of hidden vector 
        V:  dimension of vocabulary
        random_seed: random seed for consistent results in the unit tests
     Outputs: 
        W1, W2, b1, b2: initialized weights and biases
    '''
    
    np.random.seed(random_seed)

    # W1 has shape (N,V)
    W1 = np.random.rand(N,V)
    # W2 has shape (V,N)
    W2 = np.random.rand(V,N)
    # b1 has shape (N,1)
    b1 = np.random.rand(N,1)
    # b2 has shape (V,1)
    b2 = np.random.rand(V,1)

    return W1, W2, b1, b2


# softmax function
def softmax(z):
    
    e_z = np.exp(z)
    yhat = e_z/np.sum(e_z,axis=0)
    
    return yhat


# Foreward propagation to calculate H and Z2 

def forward_prop(x, W1, W2, b1, b2):
    
    z_1 = np.dot(W1, x) + b1
    
    # ReLU function ReLU(Z) = max(0, Z)
    h = np.maximum(0, z_1)
    #-------------
    z_2 = np.dot(W2, h) + b2
    
    return z_2, h

def compute_cost(y, yhat, batch_size):
    # cost function 
    logprobs = np.multiply(np.log(yhat),y) + np.multiply(np.log(1 - yhat), 1 - y)
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost


def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    
    # grad_w1 = 1/m * ReLU(W2.T . (yhat - y)) . X.T
    l1 = np.dot(W2.T,(yhat-y))
    l1 = np.maximum(0,l1)
    grad_W1 = (1 / batch_size) * np.dot(l1, x.T)
    
    # grad_w2 = 1/m * (yhat - y) . h.T
    grad_W2 = (1 / batch_size) * np.dot(yhat-y,h.T)
    
    # grad_h1 = 1/m * ReLU(W2.T . (yhat - y)) . 1.T where 1.T is a row vector with m elements all 1 
    grad_b1 = np.sum((1/batch_size)*np.dot(l1,x.T),axis=1,keepdims=True)
    
    # grad_h1 = 1/m * (yhat - y) . 1.T
    grad_b2 = np.sum((1/batch_size)*np.dot(yhat-y,h.T),axis=1,keepdims=True)
    
    return grad_W1, grad_W2, grad_b1, grad_b2



def gradient_descent(data, word2Ind, N, V, num_iters, C, alpha=0.03):
    
    W1, W2, b1, b2 = initialize_model(N,V, random_seed=282)
    batch_size = 128
    iters = 0
    C = 2
    
    for x, y in get_batches(data, word2Ind, V, C, batch_size):

        # Get z and h
        z, h = forward_prop(x, W1, W2, b1, b2)
        # Get yhat
        yhat = softmax(z)
        # Get cost
        cost = compute_cost(y, yhat, batch_size)
        if ( (iters+1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")
        # Get gradients
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)
        
        # Update weights and biases
        W1 -= alpha*grad_W1 
        W2 -= alpha*grad_W2
        b1 -= alpha*grad_b1
        b2 -= alpha*grad_b2
        
        ### END CODE HERE ###
        
        iters += 1 
        if iters == num_iters: 
            break
        if iters % 100 == 0:
            alpha *= 0.66
            
    return W1, W2, b1, b2

try:
    C = int(args['half_size'])
except Exception:
    raise Exception("Could not convert half_size to an integer, make sure that half_size value is an integer.")

try:
    N = int(args['emb_dim'])
except Exception:
    raise Exception("Could not convert emb_dim to an integer, make sure that emb_dim value is an integer.")

try:
    num_iters = int(args['num_iters'])
except Exception:
    raise Exception("Could not convert num_iters to an integer, make sure that num_iters value is an integer.")


which_w = args['emb_weight_matrix']
choose_w = ['w1', 'W1', 'w2', 'W2', 'w1w2', 'W1w2', 'w1W2', 'W1W2']
if which_w not in choose_w:
    raise Exception("emb_weight_matrix is not understood, please choose from: ['w1', 'W1', 'w2', 'W2', 'w1w2', 'W1w2', 'w1W2', 'W1W2']")



def train_extract_and_save_emb(data, word2Ind, N, V, C, num_iters, which_w):
    
    """This function takes the weights trained by the shallow NN and return the word embeddings
    is three ways:
    W1_emb: each row is an embedding for the words in the same order of words in the vocabulary
    W2_emb: each row is an embedding for the words in the same order of words in the vocabulary
    W1_W2_emb: is a combination of W1 and W2
    """
    W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, C, num_iters)
    
    if which_w == 'w1' or which_w == 'W1':
        embeddings = W1.T
        
    elif which_w == 'w2' or which_w == 'W2':
        embeddings = W2
    elif which_w == 'w1w2' or which_w == 'W1w2' or which_w == 'w1W2' or which_w == 'W1W2':
        embeddings = (W1.T + W2)/2.0
      

    # store each word and its embedding in a dictionary
    # where the keys are the words and the values are the 
    # embeddings for each word
    word_and_emb = {}

    for word, index in word2Ind.items():
    
        emb_for_word = embeddings[index, :]
        word_and_emb[word] = emb_for_word

    # save word embeddings in a pickle file
    pkl_file = open("word_emmebding.pkl", "wb")
    pickle.dump(word_and_emb, pkl_file)
    pkl_file.close()


if __name__ == "__main__":
    
   train_extract_and_save_emb(data, word2Ind, N, V, C, num_iters, which_w)