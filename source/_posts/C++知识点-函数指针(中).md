---
title: C++知识点：函数指针（中）
date: 2023-09-17
categories: 
    - C++学习日志
mathjax: true
tags: 
    - C++
description: C++知识点 函数指针进阶
top: 1
---

上一篇介绍了C++中函数指针的作用与基本使用方法，而本篇将会介绍函数指针与数组指针和指针数组结合起来的使用方法。

## 1. 将函数指针与数组指针结合使用

我们仍然从一个例子入手，假如`f1,f2,f3`是三个自定义函数，它们函数分别返回一个数组中第0，1，2个元素的地址，接收的参数为一个`const double`类型的数组，另一个为数组长度`n`（其实这三个函数内容我们不关心）。

首先定义一个长度为3的数组`av`，并初始化里面的值。

再定义一个指向函数`f1`的函数指针`const double *(*p1)(const double *, int)`，这里看起来可能有点抽象，但我们可以逐一来解析它。

首先`(*p1)`确定了**p1是一个指针**，该指针指向函数f1的地址，因为f1的返回值必须是double类型的指针，且参数必须是`const double *`和`int`，所以建立函数指针的时候也要严格按照指向函数的返回值和参数指定。此外，还可以用`auto`关键字，它能自动识别新建变量应该具备的数据类型，这里使用`auto p2`创建一个p2指针，它指向的是函数f2。

然后在调用的时候，直接使用`(*p1)(av, 3)`，将p1指针指向的内容用`*`取出即可，但是这里由于f1返回的是一个指针类型，所以如果直接打印`(*p1)(av, 3)`，将会显示数组首个元素的地址，如果想要查看值，则应对指针再进行取值操作`*(*p1)(av, 3)`。

```cpp
#include <iostream> 

using namespace std; 

const double *f1(const double *ar, int n);
const double *f2(const double *ar, int n);
const double *f3(const double *ar, int n);


int main(void){
    double av[3] = {1112.3, 1542.6, 2227.9};

    // part 1
    // pt to a function 
    const double *(*p1)(const double *, int) = f1;
    auto p2 = f2;
    cout << "-------------Part 1------------" << endl;
    cout << " Address        Value" << endl; 
    cout << (*p1)(av, 3) << ":    "<< *(*p1)(av, 3) << endl; // 等价于： f1(av, 3)
    cout << p2(av,3) << ":    " << *p2(av,  3) << endl;//等价于: f2(av,3)
}

const double *f1(const double *ar, int n){
    return ar;
}

const double *f2(const double ar[], int n){
    return ar + 1;
}

const double *f3(const double ar[], int n)
{
    return ar + 2;
}

```

输出
```
-------------Part 1------------
 Address        Value
0x61fdc0:    1112.3
0x61fdc8:    1542.6
```

## 2. 函数指针与指针数组结合使用

指针数组是指定义的数组中存放的都是指针，常见的指针数组定义方式为`*p[n]`，n为数组大小，因为中括号的优先级比*高，所以p是一个有n个元素的数组，数组中每个元素都是指针。仍然用一个例子来说明，我们在这里定义了一个指针数组`const double *(*pa[3])(const double * int)`，这个定义与第一部分中唯一的区别就是在指针pa后加了一个中括号。因此这个定义的意思是：定义了一个**数组pa**，该数组有3个元素，分别被初始化为指向函数f1，f2，f3的指针。

调用时，可以直接按照数组的形式调用指针数组中的元素，例如`pa[0](av, 3)`就等价于`f1(av, 3)`，同样，如果要取出f1函数返回地址中的值，我们需要使用额外的*来取值。

```cpp
#include <iostream> 

using namespace std; 

const double *f1(const double *ar, int n);
const double *f2(const double *ar, int n);
const double *f3(const double *ar, int n);


int main(void){
    double av[3] = {1112.3, 1542.6, 2227.9};

    // part 2
    // pa是一个指针数组，数组中每个指针指向函数f1, f2, f3
    const double *(*pa[3])(const double *, int) = {f1, f2, f3};
    auto pb = pa; 
    cout << "-------------Part 2------------" << endl;
    cout << " Address        Value" << endl; 
    for (int i = 0; i < 3; i++){
        cout << pa[i](av, 3) <<": "<< *pa[i](av, 3) << endl;;
    }
    

}

const double *f1(const double *ar, int n){
    return ar;
}

const double *f2(const double ar[], int n){
    return ar + 1;
}

const double *f3(const double ar[], int n)
{
    return ar + 2;
}
```

输出:
```
-------------Part 2------------
 Address        Value
0x61fdc0: 1112.3
0x61fdc8: 1542.6
0x61fdd0: 2227.9
```

## 3. 综合使用

我们还可以定义一个指针，这个指针指向一个指针数组，仍然用一个例子来阐述。这里我们定义了两个指针，一个使用`auto`，指向上一个例子中的数组pa，另一个比较复杂: `const *(*(*pd)[3])(const double *, int)`，我们来注逐一解析它。

首先，`(*pd)`确定了**pd是一个指针**，应该指向某个地址，然后根据符号运算优先级，这个指针指向一个包含了3个元素的数组，数组中每个元素都是指向函数的指针（这个数组本身与上一个例子中数组pa使相同的，这里只是额外定义了一个指针指向这个数组而已）。这里我们直接使pd指向pa，因此对pa取地址即可。

调用时，`(*pd)[2] == pa[2]== f2(av, 3)`，f2返回传入输入ar的第三个元素的地址。如果要查看该地址下的值，则再使用*取值即可：`*(*pd)[2] == *pa[2]`。此外，这里的`const double *pbd`定义了一个指针，该指针指向指针数组数组指针`(*pd)[1]`的地址，它等价于`pa[1]`。

```cpp
#include <iostream> 

using namespace std; 

const double *f1(const double *ar, int n);
const double *f2(const double *ar, int n);
const double *f3(const double *ar, int n);


int main(void){

    // part 3
    auto pc = &pa; //pc是一个指向一个数组的指针，这个数组每个元素都是函数指针
    const double * (*(*pd)[3])(const double *, int) = &pa; 
    cout << "-------------Part 3------------" << endl;
    cout << " Address        Value" << endl; 
    cout << (*pc)[0](av, 3) << ": " << *(*pc)[0](av,3) << endl;// (*pc) == pa --> (*pc)[0](av, 3) == pa[0](av, 3)
    const double *pbd = (*pd)[1](av, 3); // (*pd)[1](av, 3) == pa[1](av, 3) == f2(av,3)
    cout << pbd << ": " << *pbd << endl;
    cout << (*pd)[2](av, 3) << ": " << *(*pd)[2](av, 3) << endl;

}

const double *f1(const double *ar, int n){
    return ar;
}

const double *f2(const double ar[], int n){
    return ar + 1;
}

const double *f3(const double ar[], int n)
{
    return ar + 2;
}
```

输出：

```
-------------Part 3------------
 Address        Value
0x61fdc0: 1112.3
0x61fdc8: 1542.6
0x61fdd0: 2227.9
```

## 4. typedef与函数指针的结合使用

typedef作用于普通数据类型时，相当于对该数据类型进行了重命名，例如`typedef unsigned int uint`，而作用于函数指针时，代表定义了一个新的指针类型，并且我们可以用多重typedef结合函数指针来定义。举个例子，第一行typedef了一个函数指针fp_t，先不看后两行，我们可以直接使用`fp_t p`来表示一个返回值为int，接收一个int和一个char** 作为参数的函数指针，这样可以简化使用多重*带来的阅读复杂度。

```cpp
typedef int (*fp_t)(int, char**); //fp_t是一个函数指针，指向一个返回值为int，参数为int和char**的函数。这一行typedef了一个函数指针，后续可以用fp_t表示一个可以指向该类函数的指针类型。

int func(int n, char ** m);

int main(void){
    fp_t p = func;
}


int func(int n, char ** m){
    // function realization
}
```