#include<iostream>
using namespace std;

int a = 100;    //全局变量
// 局部变量优先级高
void test1(){
    int a = 10;     // 局部变量
    cout << a <<endl;
}

// :: 作用域运算符，用来访问重名的全局变量
// std::cout, std::endl同理
void test2(){
    int a = 10;
    cout << ::a<<endl;
}
int main(){
    test1();
    test2();
    system("pause");
    return EXIT_SUCCESS; 
}