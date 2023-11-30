---
title: C++知识点：函数指针（上）
date: 2023-09-17
categories: 
    - C++学习日志
mathjax: true
tags: 
    - C++
description: C++知识点 函数指针初探
top: 1
---

## 1. 函数指针的作用

在了解函数指针之前，我们先了解一下函数指针的作用。函数指针可以作为参数传递给另外一个函数，这种方法也称为回调函数。使用回调函数可以将不同实现方法的函数传入同一个调用它们的函数当中，接下来我们将会用一个例子解释这段话的含义。

## 2. 函数指针的使用

我们从一个例子入手，函数`Rick`和`Jack`是一个简单的假设估计代码运行时间的程序（真正估计代码运行时间的程序比较复杂，但这里不重要）。`estimate`接收两个参数，第一个是整形参数`lines`，代表代码的行数，`double (*pf)(int)`**代表传入一个函数指针，该指针指向的函数必须以`double`类型为返回值，并接收`int`的参数。**

在`main`函数中调用estimate时，就可以给estimate传入函数指针作为回调函数，并且这个函数指针可以指向不同的函数。在这个例子里，我们往`estimate`函数传入了Rick和Jack两个人两种估计方式，而假如是在一个项目中，我们就可以提供一个函数指针接口，让用户自己定义一个函数来计算执行代码的时间。

```cpp
#include <iostream> 

using namespace std; 

double Rick(int lines);
double Jack(int lines);
void estimate(int lines, double (*pf)(int));


int main(void){
    int code;
    cout << "How many lines of code do you need?";
    cin >> code;
    cout << "Here is Rick's estimate:" << endl; 
    estimate(code, Rick);
    cout << "Here is Jack's estimate:" << endl; 
    estimate(code, Jack);
}
//函数指针可以在同一个函数中调用不同的实现方式
double Rick(int lines){
    return lines * 0.05;
}

double Jack(int lines){
    return 0.03 * lines + 0.004 * lines;
}

void estimate(int lines, double (*pf)(int)){
    cout << lines << " lines code will take " << (*pf)(lines) << " seconds." << endl; 
}
```