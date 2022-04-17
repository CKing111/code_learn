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