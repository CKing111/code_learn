#include<iostream>
#include <stack>
using namespace std;

//CQueue创建双栈列
class CQueue {
    stack<int> stack1,stack2;
public:
    //  声明空的栈列
    CQueue() {
        while (!stack1.empty()) {
            stack1.pop();
        }
        while (!stack2.empty()) {
            stack2.pop();
        }
    }
    //push添加栈元素
    //先入stack1中，先入后出
    void appendTail(int value) {
        stack1.push(value);
    }
    // 执行队列删除操作
    //将stack1中的先入后出顺序转移到stack2中的先进先出顺序
    int deleteHead() {
        // 如果第二个栈为空
        if (stack2.empty()) {
            while (!stack1.empty()) {           //while循环，将stack1中所有元素转变到stack2中
                stack2.push(stack1.top());
                stack1.pop();
            }
        } 
        //执行删除操作
        if (stack2.empty()) {
            return -1;
        } else {
            int deleteItem = stack2.top();  // = stack1最底部元素
            stack2.pop();
            return deleteItem;  //返回删除值
        }
    }
};
