---
title: C++知识点：const与指针
date: 2023-09-06
categories: 
    - C++学习日志
mathjax: true
tags: 
    - C++
description: C++知识点 const与指针
top: 1
---

## 1. const指针的三种写法

C++种const指针有三种写法

```cpp
int const *pt = const int *pt; //第一种
int *const pt; //第二种
const int *const pt; //第三种
```

## 2. 三种const指针的区别和用例

三种const指针可以通过区分`*`和`const`的位置来确定不同的功能。简而言之，`*`在`const`之**前**为可以**修改**指向地址的**值**而不能修改指向的地址，`*`在`const`之**后**为可以**修改**指向的**地址**但不可以修改指向地址的值。如果`*`前后都有`const`则地址和值都不可以修改。

### 2.1 第一种const指针

`int const *pt;`

第一种const指针`*`在`const`之后，所以可以修改指向的地址，但不可以修改指向地址的值。

```cpp
int n = 10;
int const *pt;

pt = &n; //让pt指向n的地址
cout << "Address of pt is " << pt << endl;
*pt = 20; //不允许，报错为：表达式必须为可修改的左值

int m = 20;
pt = &m; //允许，将pt重新指向m的地址
cout << "Address of pt is " << pt << endl;
```

注释掉报错行，输出为

```
Address of pt of n is 0x61fe14
Address of pt of m is 0x61fe10
```

### 2.2 第二种const指针

`int *const pt;`

第二种const指针`*`在`const`之前，所以可以修改指向地址的值，但不可以修改指向的地址。

```cpp
int n = 10;
int *const pt = &n; //让pt指向n的地址

cout << "Value of pt is " << *pt << endl;

*pt = 20; //允许，修改n的值为20
cout << "Value of pt(modified) is " << *pt << endl;

int m = 30;
pt = &m; //不允许，报错为：表达式必须为可修改的左值
```

注释掉报错行，输出为
```
Value of pt is 10
Value of pt(modified) is 20
```

### 2.3 第三种const指针

`const int *const pt`

第三种const指针`*`前后都有`const`，所以地址和值都不可以修改。

```cpp
int n = 10;
const int *const pt = &n; //让pt指向n的地址

*pt = 20; //不允许，报错为：表达式必须为可修改的左值

int m = 30;
pt = &m; //不允许，报错为：表达式必须为可修改的左值
```

## 3. 尽可能在函数形参中使用const指针

如果不希望在函数内部修改数组的值或地址，则应该使用const指针来作为形参传参。例如

```cpp
#include <iostream>
using namespace std;

void show_array(int const *pt, int n);

int main(void){
    int size = 5;
    int arr[size] = {1,2,3,4,5};
    show_array(arr, size);
    return;
}



void show_array(int const *pt, int n){
    for (int i = 0; i < n; i++){
        cout << *pt << endl;
    }
}

```
