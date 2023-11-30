---
title: C++知识点：new/delete与malloc/free
date: 2023-09-09
categories: 
    - C++学习日志
mathjax: true
tags: 
    - C++
description: C++知识点 new/delete与malloc/free的联系与区别
top: 1
---



## 1. new与malloc的用法

C++中new的作用与C语言中malloc的作用类似，都是在堆(heap)申请一块内存空间。对于基本数据类型，如int, char等，二者的用法相同，例如

```cpp
int n; // array length
int *p_m = (int *)malloc(sizeof(int)*n);
int *p_n = new int [n];
free(p_m)
delete [] p_n
```

**而在创建类的时候，只能用new而不能用malloc**。例如

```cpp
class A{
    public:
        int a;
        int b;
        int plus(int a,int b);
}

int A::plus(int a, int b){
    return a+b;
}

int main(void){
    A *p = new A();
    p->a = 10;
    p->b = 20;
    int add = p->plus(p->a, p->b)
    delete p;
}

```

另外，new后面可接`()`或`[]`：
- 接`()`表示只申请一个该类型的内存空间，并用括号内的值对其进行初始化
- 接`[]`表示申请多个该类型的内存空间，中括号内的值为申请类型的个数

例如
```cpp
int *p1 = new int(10);
/* 等效于
int *p1 = new int();
*pt = 10;
*/
int *p2 = new int[3];

delete p1;
delete p2;
```

## 2. new/delete和malloc/free的区别

引用自：[【C++】C++ new和malloc到底哪里不一样](https://www.cnblogs.com/lcgbk/p/14118782.html)

### 2.1 属性上的区别

new/delete是关键字，使用时需要编译器支持
malloc/free是库函数，使用时需要include对应的库

### 2.2 使用上的区别

malloc需要显式传入申请内存空间的大小，而new会自动根据申请的类型分贝大小。例如

```cpp
int *p_m = (int *)malloc(sizeof(int));
int *p_n = new int();
```

### 2.3 存储位置不一样

new：申请空间为自由储存区
malloc：申请空间为堆

**堆** 是C语言和操作系统的术语，堆是操作系统所维护的一块特殊内存，它提供了动态分配的功能，当运行程序调用malloc()时就会从中分配，调用free()归还内存。

**自由储存区** C/C++的内存通常分为：堆、栈、自由存储区、全局/静态存储区、常量存储区。其中自由储存区可以是堆、全局/静态储存区等。但具体是哪一个取决于new的实现和C++默认的new分配空间的位置。但是基本上C++默认的new分配空间都在堆上。最后new/delete关键字是可以被重载的，也就是说我们可以修改new分配的内存空间位置，而malloc/free是C库中的函数，是无法被重载的。

### 2.4 返回类型不同

new直接返回一个指定的数据类型或者对象的指针，而malloc默认返回的是一个void *，需要通过强制类型转换变成需要的类型。因此，new符合类型安全性的操作符，比malloc更加可靠。

### 2.5 分配失败

malloc分配失败会返回NULL指针
new分配失败会抛出bad_alloc_异常，我们可以通过异常捕获的方式获取该异常。

### 2.6 定义对象系统调度过程的区别

使用new操作符来分配对象内存时会经历三个步骤：

1. 调用operator new 函数（对于数组是operator new[]）分配一块足够的内存空间（通常底层默认使用malloc实现，除非程序员重载new符号）以便存储特定类型的对象；

2. 编译器运行相应的构造函数以构造对象，并为其传入初值。

3. 对象构造完成后，返回一个指向该对象的指针。

使用delete操作符来释放对象内存时会经历两个步骤：

1. 调用对象的析构函数。

2. 编译器调用operator delete(或operator delete[])函数释放内存空间（通常底层默认使用free实现，除非程序员重载delete符号）。

## 2.7 扩张内存大小的区别

malloc：使用malloc分配内存后，发现内存不够用，那我们可以通过realloc函数来扩张内存大小，realloc会先判断当前申请的内存后面是否还有足够的内存空间进行扩张，如果有足够的空间，那么就会往后面继续申请空间，并返回原来的地址指针；否则realloc会在另外有足够大小的内存申请一块空间，并将当前内存空间里的内容拷贝到新的内存空间里，最后返回新的地址指针。
new：new没有扩张内存的机制。

### 2.8 总结

|        特征        |            new/delete            |              malloc/free             |
|:------------------:|:--------------------------------:|:------------------------------------:|
| **分配内存的位置**     | 自由存储区                       | 堆                                   |
| **内存分配失败**       | 抛出异常                         | 返回NULL                             |
| **分配内存的大小**     | 编译器根据类型计算得出           | 显式指定字节数                       |
| **处理数组**           | 有处理数组的new版本new[]         | 需要用户计算数组的大小后进行内存分配 |
| **已分配内存的扩张**   | 不支持                           | 使用realloc完成                      |
| **分配内存时内存不足** | 可以指定处理函数或重新制定分配器 | 无法通过用户代码进行处理             |
| **是否可以重载**       | 可以                             | 不可以                               |
| **构造函数与析构函数** | 调用                             | 不调用                               |