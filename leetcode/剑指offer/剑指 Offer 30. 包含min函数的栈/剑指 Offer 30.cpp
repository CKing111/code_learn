#include<stack>
#include<iostream>
using namespace std;

//1
class MinStack {
    stack<int> x_stack; //主栈
    stack<int> min_stack;   //辅助栈
public:
    MinStack() {
        min_stack.push(INT_MAX);
    }
    
    //入栈
    //
    void push(int x) {
        x_stack.push(x);
        min_stack.push(std::min(min_stack.top(), x));    //入栈值需要于栈头比较，压小
    }
    
    //出栈
    void pop() {
        x_stack.pop();
        min_stack.pop();
    }
    
    int top() {
        return x_stack.top();
    }
    
    int min() {
        return min_stack.top();
    }
};

//2
class MinStack {
private:
    stack<int>valStack; //主栈
    stack<int>minStack; //辅助栈
public:
    MinStack(){ //初始化
        while(!valStack.empty())valStack.pop();
        while(!minStack.empty())minStack.pop();
    }
    //入栈
    void push(int x) {
        minStack.push(minStack.empty()?x:std::min(minStack.top(),x));   //每次只入最小值，因此栈数量于主栈相同
        valStack.push(x);
    }
    
    void pop() {
        valStack.pop();
        minStack.pop();
    }
    
    int top() {
        return valStack.top();
    }
    
    int min() {
        return minStack.top();
    }
};
