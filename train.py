import numpy as np
import random
import math
import time
import scipy.sparse
import multiprocessing
def rowM(matrix,M):
    B = matrix[:M]
    return B
def Mrow(matrix,M):
    B = matrix[:M]
    B = B.transpose()
    print (B.shape)
    return B

def topK(vector_origin,vector_propagate,M,N,k):
    recommendList = []
    recommend_vector = [np.zeros(N) for _ in range(M)]
    vector = vector_propagate - 1000*vector_origin
    print(type(vector_origin),vector_origin.shape,type(vector_propagate),vector_propagate.shape)
    for user in range(M):
        print("topK of user",user)
        sorted_indices = np.argsort(vector[user])
        topk_indices = sorted_indices[-k:]
        for idx in topk_indices:
            recommend_vector[user][idx] = 1
            recommendList.append((user,idx))
        print("user",user,"finished")
    return recommendList, recommend_vector

def evaluate(recommendList, test):
    count = 0
    count2 = 0
    print("Evaluating...")
    RecLenth = len(recommendList)
    for tuple_item in recommendList:
        count2 +=1
        user = tuple_item[0]
        item = tuple_item[1]
        # testnp = numpy_array = np.array(test)
        for test_item in test[user]:
            if (test_item == item):
                count += 1
                break
        print("Evaluating:",count2,"/",RecLenth)
    return count