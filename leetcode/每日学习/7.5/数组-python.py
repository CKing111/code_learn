
#https://leetcode.cn/leetbook/read/top-interview-questions-easy/x2gy9m/
# 题目1：删除排序数组中的重复项

"""
给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。

由于在某些语言中不能改变数组的长度，所以必须将结果放在数组nums的第一部分。更规范地说，如果在删除重复项之后有 k 个元素，那么 nums 的前 k 个元素应该保存最终结果。

将最终结果插入 nums 的前 k 个位置后返回 k 。

不要使用额外的空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

判题标准:

系统会用下面的代码来测试你的题解:

int[] nums = [...]; // 输入数组
int[] expectedNums = [...]; // 长度正确的期望答案

int k = removeDuplicates(nums); // 调用

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
如果所有断言都通过，那么您的题解将被 通过。

 

示例 1：

输入：nums = [1,1,2]
输出：2, nums = [1,2,_]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
示例 2：

输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]
解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。
 

提示：

1 <= nums.length <= 3 * 104
-104 <= nums[i] <= 104
nums 已按 升序 排列


"""
# 初稿
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        res = []
        num_len = len(nums)
        for i in range(num_len):
            x = nums[i]
            if x not in res:
                res.append(nums[i])
            else:
                nums.remove(i)
        return nums


#思考
# 题解1
#   （1）双指针法
# 使用两个指针，右指针始终往右移动，

# 如果右指针指向的值等于左指针指向的值，左指针不动。
# 如果右指针指向的值不等于左指针指向的值，那么左指针往右移一步，然后再把右指针指向的值赋给左指针。

#   
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        index = 0  # 锚点位置
        for i in range(1, len(nums)):
            flag= nums[index]
            if flag >= nums[i]: # 左指针值大于等于右指针值
                continue
            else:       # 左右指针值不相等
                index += 1   # 锚点右移（左指针右移）
                nums[index] = nums[i]  # 锚点所在位置更新为找到的元素
                flag = nums[index]  # 更新锚点元素

        return index + 1

#   优化
# 当数组 nums 的长度大于 0 时，数组中至少包含一个元素，在删除重复元素之后也至少剩下一个元素，因此 nums[0] 保持原状即可，从下标 1 开始删除重复元素。
# 定义两个指针 fast 和 slow 分别为快指针和慢指针，快指针表示遍历数组到达的下标位置，慢指针表示下一个不同元素要填入的下标位置，初始时两个指针都指向下标 1。


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        fast = slow = 1 # 初始化双指针
        while fast < n: # 遍历快指针 1到（n-1）
            if nums[fast] != nums[fast - 1]:    # 表示排序好的数组出现变化
                nums[slow] = nums[fast]         # 更新慢指针数据
                slow += 1                       # 后移一位慢指针
            fast += 1                           # 后移一位快指针
        
        return slow


# 题目2：买卖股票的最佳时机 II
#https://leetcode.cn/leetbook/read/top-interview-questions-easy/x2zsx1/

"""
给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。

返回 你能获得的 最大 利润 。

 

示例 1：

输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
     总利润为 4 + 3 = 7 。
示例 2：

输入：prices = [1,2,3,4,5]
输出：4
解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     总利润为 4 。
示例 3：

输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 交易无法获得正利润，所以不参与交易可以获得最大利润，最大利润为 0 。
 

提示：

1 <= prices.length <= 3 * 104
0 <= prices[i] <= 104

"""

# 解题1：动态规划
# 贪心算法
