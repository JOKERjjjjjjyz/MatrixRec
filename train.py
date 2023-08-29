import numpy as np
import random
import math
import time
import scipy.sparse
def randomwalk(length,graph,start_node):
    current_node = start_node
    for step in range(length):
        # 获取当前节点的邻居节点
        neighbors = graph[current_node].indices
        if len(neighbors) == 0:
            # 当前节点没有邻居，随机游走结束
            break
        # 随机选择一个邻居作为下一步的节点
        next_node = random.choice(neighbors)
        # 更新当前节点为下一步的节点
        current_node = next_node
    radio = 1/length
    return current_node,radio

def rowM(matrix,M):
    B = matrix[:M]
    return B
def Mrow(matrix,M):
    # 假设 A 是一个 N*N 的 CSR 矩阵，M 是你想要提取的行数
    # 提取前 M 行的数据
    # N = matrix.shape[0]
    # print(N)
    # print(matrix.indptr,matrix.data,matrix.indices)
    # B_data = matrix.data[:M]
    # B_indices = matrix.indices[:M]
    # B_indptr = matrix.indptr[:M + 1]
    #
    # print(B_indices,B_indptr)
    # # 构建 CSR 矩阵 B
    # B = scipy.sparse.csr_matrix((B_data, B_indices, B_indptr), shape=(M,N))
    B = matrix[:M]
    B = B.transpose()
    print (B.shape)
    return B

def propagate(k,graph,vector_origin,M,N,KsampleNum):
    user_list = [i for i in range(M)]
    vector = np.zeros((M + N, N))
    for user in user_list:
        start_time=time.time()
        for j in range(KsampleNum):
            print("Training:Epoch",k,",(user,j):(",user,",",j,")")
            targetNode,radio = randomwalk(k, graph, user)
            vector[targetNode] += radio*vector_origin[user]*0.001
        end_time = time.time()
        use_time = end_time - start_time
        print("user",user,"'s time:",use_time,"s")
    return vector

def Klayer_sampleNum(k,epsilon,delta,M,index):
    # return N: sample number for k
    N = 1/(2*epsilon*epsilon)*math.log(2*M/delta)*M*math.pow(k,index)
    return int(N)+1

def topK(vector_origin,vector_propagate,M,N,k):
    print("here")
    recommendList = []
    print("here")
    recommend_vector = [np.zeros(N) for _ in range(M)]
    print(type(vector_origin),vector_origin.shape,type(vector_propagate),vector_propagate.shape)
    for user in range(M):
        print("topK of user",user)
        recommend_vector = vector_propagate - 10000*vector_origin
        sorted_indices = np.argsort(vector_propagate[user])
        # 获取 top-k 大值的索引
        topk_indices = sorted_indices[-k:]
        for idx in topk_indices:
            recommend_vector[user][idx] = 1
            recommendList.append((user,idx))
        # user_recommendList[v] = maximizeK(vector_propagate)
        # for item in user_recommendList[v] do: recommendList.append({v,item})
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

# def main(graph,vector_origin)
#     # M:user number; N: item number
#     # vector_origin: M*N;  vector_propagate: (M+N)*N
#     vector_propagate = [0]
#     for i:1 to K do:
#         vector_propagate = propagate(i,graph,vector_origin,vector_propagate)
#     topK(vector_origin,vector_propagate)
#     evaluate(recommendList , test)
#     return 0