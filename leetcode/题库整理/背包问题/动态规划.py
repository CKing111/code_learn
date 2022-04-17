#



#给定一个数组，求出递增子序列最大元素个数

#暴力枚举

def L(nums,i):
    """返回起始于i的递增子序列长度个数，num为输入数组，i为起始数组"""

    if i == len(nums) - 1:   #last number
        return 1

    max_len = 1     #初始加上自身个数
    for j in range (i + 1, len(nums)):      #历遍0起始数组
        if nums[j] > nums[i]:
            max_len = max(max_len,L(nums,j)+1)      #历遍所有nums数组值与初始i值进行比较，当小于初始时返回自身，大于初始时返回自身加1
        return max_len

def Length_of_LST(nums):
    max_len = 1
    for i in range(len(nums)):
        max_len = max(max_len,L(nums,i)+1)             #将上述操作历遍数组
    return max_len

nums = [1,5,2,4,3]

print(Length_of_LST(nums))


#问题复杂度较高


# 采用动态规划，通过避免重复节点的运算，来加速整个计算过程
# 采用字典、哈希表来存储中间计算过程
# 记忆化搜索：拿空间换时间，也称带备忘录的递归、递归树的剪枝(pruning)


memo = {}       #声明一个字典，来记录过程中计算出来的结果
def L(nums,i):
    """返回起始于i的递增子序列长度个数，num为输入数组，i为起始数组"""

    if i in memo:
        return memo[i]

    if i == len(nums) - 1:   #last number
        return 1

    max_len = 1     #初始加上自身个数
    for j in range (i + 1, len(nums)):      #历遍0起始数组
        if nums[j] > nums[i]:
            max_len = max(max_len,L(nums,j)+1)      #历遍所有nums数组值与初始i值进行比较，当小于初始时返回自身，大于初始时返回自身加1
    memo[i] = max_len
    return max_len

def Length_of_LST(nums):
    max_len = 0
    for i in range(len(nums)):
        max_len = max(max_len,L(nums,i))             #将上述操作历遍数组
    return max_len

nums = [1,5,2,4,3]

print(Length_of_LST(nums))


#非递归方法：迭代(iterative)计算形式

"""
[1,5,2,4,3]
计算公式，从1出发以此计算后面参数：
L(0) = max{L(1),L(2),L(3),L(4)} + 1
L(1) = max{L(1),L(2),L(3)} + 1      #从5出发
L(2) = max{L(1),L(2)} + 1
L(3) = max{L(1)} + 1
L(4) = 1

从后往前以此计算，类似于数学迭代法
实现：
"""

def Length_of_LST(nums):
    n = len(nums)   #5
    L = [1] * n     #初始化列表L:[1，1，1，1，1]

    for i in reversed(range(n)):          #循环列：由L(4) -> L(0)
        for j in range(i+1,n):            #循环行：由0 -> max{L(1),L(2),L(3),L(4)}
            if nums[j] > nums[i]:         #构成递增子序列
                L[i] = max(L[i],L[j] +1)

    return max(L)

    #上式复杂度为O(N^2)小于之前为O(2^N)


#题目：找出连续子序列的最大和
"""
如：[3,-4,2,-1,2,6,-5,4]的最大子序列和为[2,-1,2,6]和为9
"""

#思路：