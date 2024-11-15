import numpy as np

data = np.array(
    [
        [5, 3, 4, 4, -1],  # Alice, 物品5为未知量-1
        [3, 1, 2, 3, 3],  # user 1
        [4, 3, 4, 3, 5],  # user 2
        [3, 3, 1, 5, 4],  # user 3
        [1, 5, 5, 2, 1],  # user 4
    ]
)

data_transpose = data.T
sim_mat = np.corrcoef(data[1:, :].T)


# argsort是从小到大，这里需要取反，从大到小
similar_item = np.argsort(-sim_mat[-1]) 
print(similar_item)
# 最大的一定是1，也就是自己，从第二大开始取
num = 2
similar_item = similar_item[1:num+1]
print(similar_item)
# 获取相似用户的评分
similar_item_data = np.array([data_transpose[i, :] for i in similar_item])
print(similar_item_data)
# 获取相似用户的相似度
similar_item_sim = np.array([sim_mat[-1, i] for i in similar_item])
print(similar_item_sim)
# 计算相似用户均值
similar_item_means = [np.mean(similar_item_data[i, :]) for i in range(len(similar_item_data))]
print(similar_item_means)


item5_mean = np.mean(data_transpose[-1,1:])

# 计算公式
score = item5_mean + sum([similar_item_sim[i]*(similar_item_data[i, 0]-similar_item_means[i]) for i in range(len(similar_item))])/sum(similar_item_sim)



print(score)
