class CQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    #定义添加元素操作
    def appendTail(self, value: int) -> None:
        # # 当输入为stack1时，获取此时stack2的值
        # while self.stack1:
        #     self.stack2.append(self.stack1.pop())

        # 将value加入到stack1
        self.stack1.append(value)
        
        # # 当输入为stack2时，需要转移到stack1中存储
        # while self.stack2:
        #     self.stack1.append(self.stack2.pop())
        # return self.stack1

    #定义删除元素操作
    def deleteHead(self) -> int:    #返回int
        #当目标为stack2，且不为空，返回stack2头元素
        if self.stack2: return self.stack2.pop()

        if not self.stack1: return -1   #当目标为stack1且为空，返回-1

        #当目标为stack1且不为空，要想达到先进先出，需要使用两个栈转移
        while self.stack1:
            self.stack2.append(self.stack1.pop())   #stack1 -> stack2
        return self.stack2.pop()    #返回stack2头元素   