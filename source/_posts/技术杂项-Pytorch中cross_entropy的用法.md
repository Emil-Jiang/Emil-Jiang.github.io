---
title: Pytorch中Cross Entropy的用法
date: 2023-06-01
categories: 
    - 技术杂项
mathjax: true
tags: 
    - Pytorch, 深度学习
description: 本文主要阐述了Pytorch中Cross Entropy的用法。
top: 1
---

**本文假设读者已经对熵（Entropy）的概念有所了解。**

## 1. 交叉熵

本文只对交叉熵做简要介绍。详细步骤请参阅各大机器学习书籍、博客等。

交叉熵（Cross Entropy）是深度学习中常用的损失函数。交叉熵是KL散度在特定情况下的变种。KL散度公式如下，$H(p)$为数据集的概率分布熵，$H(p,q)$即为分布$p,q$的交叉熵。

$$
D(p||q) = H(p,q) - H(p) = -\frac{1}{N}\sum_{i=0}^Kp(x_i)\log[\frac{q(x_i)}{p(x_i)}]
$$

深度学习中，原概率分布$p(x)$往往是已知的，所以$H(p)$为常数，而$q(x)$为模型的logits经过softmax之后的结果，所以优化KL散度等于交叉熵本身，交叉熵公式如下。

$$
L = H(p,q) = -\frac{1}{N}\sum_{i=0}^Kp(x_i)\log[q(x_i)]
$$

## 2. Pytorch中的nn.CrossEntropyLoss()

Pytorch中的nn.CrossEntropyLoss()一般输入两个参数：output和ground truth，output一般为模型的输出，而ground truth有多种表现形式。下面举几个例子。

### 2.1 one-hot编码型

交叉熵一般用于分类任务，对于多分类任务，可以进行one-hot编码。我们直接来看代码示例。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                  [0.2, 0.3,0.35, 0.15],
                  [0.28, 0.32, 0.2, 0.3]], dtype=torch.float32, requires_grad=True) 

y = torch.tensor([[0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0]], dtype=torch.float32, requires_grad=False)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(x,y)

print(loss)
```

输出为

```
tensor(1.2914, grad_fn=<DivBackward1>)
```

这里直接将one-hot编码之后的$y$作为参数输入loss_fn，结果是可执行的。

### 2.2 索引输入型

除了直接输入one-hot编码形式的GT，我们还可以直接输入one-hot编码后$y$中每一个维度中值为1的索引。见如下示例。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                  [0.2, 0.3,0.35, 0.15],
                  [0.28, 0.32, 0.2, 0.3]], dtype=torch.float32, requires_grad=True)

#注意y的数据类型必须为torch.long()
y = torch.tensor([3, 2, 1], dtype=torch.float32, requires_grad=False).long()

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(x,y)

print(loss)
```

同样，输出结果为
```
tensor(1.2914, grad_fn=<DivBackward1>)
```

对比两段代码，数组$[3,2,1]$即为第一段代码中$y$中每一行值为1的索引。

## 3. 应用

写这个博客的原因是学习MoCo时发现计算CrossEntropyLoss时直接将所有label置零，感到有些不解，于是查阅资料发现Pytorch中的CrossEntropyLoss有多种输入方法，于此记录。

MoCo中可以直接将label置零的原因是在伪代码中有这样一行

```
logits = cat([l_pos, 1_neg], dim=1)
```

也就是说MoCo始终将正样本放在第一个位置，因此one-hot编码时值为1的地方都是首位，直接将label全部置零，用索引的形式输入CrossEntropyLoss即可。