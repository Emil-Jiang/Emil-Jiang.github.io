---
title: 论文笔记-Swin Transformer：Hierarchical Vision Transformer using Shifted Windows
date: 2023-04-15
categories: 
    - 论文笔记
mathjax: true
tags: 
    - 读论文
description: AI经典论文笔记-Swin Transformer：Hierarchical Vision Transformer using Shifted Windows
top: 1
---

论文链接：https://arxiv.org/abs/2103.14030

## 1. 摘要

Swin Transformer一文的主要目的是构造一个可以作为视觉任务通用的backbone架构。之前的ViT已经在图像分类上取得了优秀的结果，而Swin Transformer的诞生使得诸如目标检测、实例分割等下游视觉任务应用Transformer成为可能。要将Transformer运用到下游视觉任务主要面临两个挑战，第一个便是在ViT中也提到的图像序列过长的问题，第二个是物体的多尺度问题。尺度问题是指同一物体在不同场景的大小、形状或者相对位置都有可能不同，这个问题在NLP任务中是不存在的。因此，原文提出了一种与滑动窗口（**S**hifted **win**dows, Swin）结合的Transformer结构。滑动窗口借鉴了CNN的思想，首先将self-attention计算限制在一些不重合（no-overlapping）的区域来降低序列长度，同时通过shift操作使用cross-window connection来学习全局信息。另外，原文提出的hierachical结构使得模型复杂度随着图像分辨率增加而线性增加（ViT为平方增加）。

## 2. 导论

Swin Transformer的导论前两段与ViT差不多。导论的核心内容是与ViT在多尺度问题上做了对比。在阐述原文的对比之前，先回顾一下CNN的多尺度特征假设：CNN可提取多尺度特征的主要原因是有多层卷积和池化操作，每一个卷积层的感受野（receptive field）是不同的，并且如果在多层的CNN结构中加入池化操作，越到后面卷积核的感受野将会越大，因此CNN能通过靠后层的卷积操作学习到多尺度特征，例如在目标检测任务中，特征金字塔（Feature pyramid network, FPN）就是用不同卷积层输出的不同level的特征图来学习多尺度特征。总之，对于下游视觉任务，多尺度问题是至关重要的。

ViT产生的特征图都是单一尺寸且是低分辨率的（下采样16倍），因此它并不适合做密集预测型任务。并且ViT始终是在全局图像上做self-attention计算，复杂度随图像分辨率增大而呈现$O(n^2)$的增长趋势，而在下游视觉任务中，图像分辨率往往会到$800\times 800$甚至$1000\times 1000$，序列长度是一个难以解决的问题。

Swin Transformer为了解决上述问题，借鉴CNN的思想，提出了滑动窗口结构，在每一个local window内做self-attention，引入了CNN中locality的这一归纳偏置（inductive bias）。另外还模仿CNN中的池化操作，提出了patch merging操作，将多个相邻的小patches合成大的patch，相当于做下采样（原文中做的2x下采样）。下图表示了Swin Transformer和ViT生成特征图的区别，可以看出，通过patch merging的操作可以生成不同尺度的特征图，将这些不同level的特征图输入到FPN或者UNet就可以做目标检测或者实力分割了。

<div align=center><img src="论文笔记-Swin Transformer/swin and vit.png" alt="Swin Transformer和ViT生成特征图的区别" ></div>

Swin Transformer的核心在于它的滑动窗口（shifted windows）结构。下图展示了shifted windows的过程，假设原图用蓝色框表示，那么在第$l$层可以将图像分为$2\times 2$的四宫格，$l$层的4个窗口分别再自己的local window内做self-attention操作，因此无法注意到其他patches的信息。而在$l+1$层将蓝色框向右下移动两个patches的距离，再划分为一个四宫格，最后就能得到右边的图中的结果了。此时由于重新划分了窗口，原先的patches归属到了不同的窗口，因此使用Swin后窗口之间可交互，到最后几层时每个patch的感受野就会变得很大了。注意到移动后有部分窗口会产生空白，一种简单的方式是直接将这些地方填充为0，但这就将原先的4个窗口变成了9个窗口，增加了计算量，因此作者提出了一种很巧妙的掩码操作，在后面会详细介绍。

<div align=center><img src="论文笔记-Swin Transformer/shifted windows.png" alt="滑动窗口" ></div>

## 3. 方法

### 3.1 模型概览

下图展示了Swin Transformer的整体模型架构，跟Attention is all you need一文的笔记一样，这里先对整体进行描述。仍然假设输入图像为$224\times 224$，原文中一个patch的大小为$4\times 4$，则共有$(\frac{224}{4})^2=56^2$共3136个子图块，每个patch为$4\times 4\times 3=48$维，那么原图就变为了$56\times 56\times 48$的一个张量，这就是模型的输入。

Stage 1中的Linear embedding层原理与ViT相同，不过这里将输入投影为到了96维，经过Linear embedding并展平后输入向量为$3136\times 96$。到这里其实跟ViT是完全相同的，但ViT的输入序列长度只有$16^2+1=197$（算上class token），3136的输入序列对于Transformer过长，因此Swin Transformer使用了带窗口的self-attention，如图“滑动窗口”一图中左侧，但原文将一个输入分为了8个窗口，每个窗口共有$7\times 7=49$个patches，那么每个窗口内的self-attention输入序列就为49，这对一个self-attention模块来讲是完全可以接受的。然后经过Swin Transformer的block，由于self-attention并不会改变维度，所以输出仍然为$56\times 56\times 96$。

Stage 2，3，4都是首先经过一个patch merging操作再连接一个Swin Transformer block，现在暂时可以先将patch merging想象成一个池化层，目的是为了增大self-attention的感受野。

<div align=center><img src="论文笔记-Swin Transformer/swin model.png" alt="Swin Transformer模型概览" ></div>

### 3.2 Patch merging
Patch merging是借鉴CNN池化的思想，用于增大感受野。Patch merging的过程如下图所示。假设输入为通道$C=1$的张量，首先将输入做2x下采样，得到$\frac{H}{2}\times \frac{W}{2}$尺寸的下采样图，然后按照得到的下采样图数量按通道的维度堆叠起来，如果做2x下采样，那么输出通道为$4C$，此时的张量尺寸为$\frac{H}{2}\times \frac{W}{2}\times 4C$，但CNN中池化之后通道数一般只会翻2倍，所以需要再做一个$1\times 1$的卷积将其映射为$\frac{H}{2}\times \frac{W}{2}\times 2C$。

Patch merging的作用是缩小特征图分辨率，调整通道数，最终形成层次化设计（hierachical design）。与CNN的池化操作不同的是，虽然patch merging增加了计算量，但并不会丢失特征图的任何信息。

<div align=center><img src="论文笔记-Swin Transformer/patch merging.png" alt="Patch merging" ></div>

### 3.3 Shifted window based self-attention

#### 3.3.1 Computation amount

用一句话表述基于窗口的self-attention就是：在划分的每个local window内做self-attention。原文设定每个窗口为$7\times 7$大小，则一幅图可分为$56/7=8$个窗口。这样做的好处是可以减少计算量，原文中给出了如下两个公式，MSA代表ViT中多头注意力机制的计算量，W-MSA代表Swin Transformer中MAS with windows的计算量。

$$
\begin{aligned}
    \Omega(MSA) = 4hwC^2+2(hw)^2C, \\
    \Omega(W-MSA) = 4hwC^2+2M^2hwC
\end{aligned}
$$

普通的attention计算维度变化如下图所示。第一阶段，计算$Q,K,V$矩阵是原张量乘以3个$C\times C$的稀疏矩阵，则计算量为$3hwC^2$。之后计算$Q,K$相似度和与$V$加权运算共计计算量为$2(hw)^2C$，最后做投影还有一个计算量为$hwC^2$的矩阵乘法。全部加起来即为MSA的计算量。

W-MSA的attention计算全部在窗口里，每个窗口有M个patches，那么MSA计算量公式中的$hw$就可以替换成$M\times M$，再套进MSA的公式中，再乘以窗口数量，即为$\Omega(W-MSA)$的公式，如下式所示，化简即为原文中给出的计算量公式。虽然看起来差别不大，但这里的$M$和$hw$很可能不在一个数量级以内，例如$hw=56^2$的话，那么$M^2=49$，差距很大。

$$
\Omega(W-MSA)=(\frac{h}{M}\times \frac{w}{M})(4M^2C^2+2M^4C)
$$

<div align=center><img src="论文笔记-Swin Transformer/attention dim.png" alt="Attention dimension flow" ></div>


#### 3.3.2 Shifted windows
滑动窗口是整篇文章最核心的思想。通过滑动窗口，Transformer就可以像CNN一样得到图像的相邻特征，相当于人为给Transformer加入了CNN中locality这一归纳偏置。

首先应当指出，只在local window内做self-attention计算是无法得到相邻特征的，也就无法进行全局建模。因为不同窗口内patches是相互独立，无法进行attention操作的。因此每一个Swin Transformer block内除了W-MSA操作，还要连接一个SW-MSA（Shifted Window-MSA）模块来保证Transformer能学到不同窗口内的邻接特征，如下图所示。每个block由两个修改过的attention模块串联组成，每个attention模块内除了将MSA模块改为W-MSA和SW-MSA外其他操作均一样。

<div align=center><img src="论文笔记-Swin Transformer/swin block.png" alt="Swin Transformer block" ></div>

Swin Transformer中的滑动窗口操作如第2节图“滑动窗口”所示，每一次向右下角滑动两个patch后再重新分配窗口。这样做的问题是每个窗口大小不相同，也就不能压成一个batch。一个简单的方法是直接在空白区域pad上0，但这样就把原先的4个窗口增加到了9个窗口，计算量提高了。

原文提出了一种非常巧妙地掩码方法，既可以保证每个batch内只有4个窗口，又可以保证每个窗口内的patches数量相同，过程如下图所示。原文首先将窗口移位之后没有被划进窗口内的patches移动填补到空白区域。这样就可以保证每一部分都是有效信息且每个窗口内的patches数量一致。但这也会引入一个新的问题，移位之后的图像的相对位置相较于原来的位置有所变化，把他们强行与图中其他区域一起做self-attention操作是不合理的。例如，一幅图中A部分原本为天空，C部分为地面，经过移位之后，天空与地面就连接在一起了，这显然是不符合认知的，所以要进行掩码操作。

<div align=center><img src="论文笔记-Swin Transformer/mask.png" alt="Shifted window" ></div>

Swin Transformer的GitHub仓库issue 38中也提出了有关mask的问题，原作者给出了如下的可视化移位之后的窗口图。直接看图可能有一些模糊，这里也借用b站up主大神Bryanzhu的讲解。以图片左下角window2为例，window2中上半部分为原图中的3，下半部分为从其他地方移位过来的6，将window2中展平为向量和它做self-attention操作可以表示为自己和自己的转置相乘（注意这里3和6表示矩阵内元素，不表示具体数值），得到的结果可以表示为一个分块矩阵。现在我们只需要33和66的部分的attention，而不需要36和63的部分，因为36个63这两部分来自图像不同位置，不需要self-attention操作得到的值。具体操作为在得到的分块矩阵上加上一个掩码矩阵，这个掩码矩阵在需要的地方填充0，不需要的地方填充为-100（或者一个很大的负数），这样在softmax操作的时候不需要的部分被放到指数上就趋近于0了。下式为window2的掩码操作，其他windows的操作相同，只是根据self-attention的结果变换一下掩码矩阵0和负数的位置即可。计算后还要做一个反向移位，不然整个图像随着层数的递增会一致往右下角移动，这样会破坏语义信息。

$$
\left[\begin{array}{c|c}
0&-100\\
\hline
-100&0
\end{array}\right]+\left[
\begin{matrix}
 3\\
 3\\
 ...\\
 3\\
 6\\
 6\\
 ...\\
 6
\end{matrix}
\right]\times \left[3,3,...,3,6,6,...,6\right] =  \left[
 \begin{array}{ccc|ccc}
   33 & ... & 33 & 36 & ... & 36 \\
   \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
   33 & ... & 33 & 36 & ... & 36 \\
   \hline
   63 & ... & 63 & 66 & ... & 66 \\
   \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
   63 & ... & 63 & 66 & ... & 66 \\
  \end{array}
  \right]+\left[\begin{array}{c|c}
0&-100\\
\hline
-100&0
\end{array}\right]
$$

<div align=center><img src="论文笔记-Swin Transformer/mask-git.png" alt="Mask operation" ></div>

## 4. 评论

Swin Transformer可以被认为是套了Transformer外套的CNN，因为它将CNN中两条重要的归纳偏置加入了Transformer中，使得Transformer可以提取到相邻特征，增大感受野。

Swin Transformer不仅仅是可以用来做图像分类，原作者更希望它能成为一个CV领域通用的backbone，我们完全可以将Swin Transformer当成一个特征提取器，然后自己设计抽头。到这里，NLP与CV模型大一统的进程又向前推进了一步。