
/*
给定一个整数数组 nums?，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例 1：
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组?[4,-1,2,1] 的和最大，为?6 。

示例 2：
输入：nums = [1]
输出：1

示例 3：
输入：nums = [0]
输出：0

示例 4：
输入：nums = [-1]
输出：-1

示例 5：
输入：nums = [-100000]
输出：-100000
*/

#include<iostream>
#include<vector>
using namespace std;


//暴力检索
class Solution
{
public:
    int maxSubArray(vector<int> &nums)
    {
        //类似寻找最大最小值的题目，初始值一定要定义成理论上的最小最大值
        int max = INT_MIN;
        int numsSize = int(nums.size());        //获取长度
        for (int i = 0; i < numsSize; i++)          //历遍
        {
            int sum = 0;                            //初始化和结果
            for (int j = i; j < numsSize; j++)      //从当前项开始历遍求和
            {   
                sum += nums[j];                     //求sum
                if (sum > max)                      //对比大小
                {
                    max = sum;                      //当求和项大于当前max，赋值
                }
            }
        }

        return max;
    }
};


//动态规划
class Solution {
public:
    int maxSubArray(vector<int>& nums) 
    {
        //初始化，默认最小值
        int r = nums.size();
        int Max = INT_MIN;
        //声明f向量，f[i]表示在第i项最大子序和2
        vector<int> f(r,0);
        //赋首值
        f[0] = nums[0];
        Max = f[0];                                 //Max为我们最后要输出的最大子序和结果，先赋初始值
        for(int i = 1;i < r;i++){                   //历遍
            f[i] = max(f[i-1]+nums[i],nums[i]);     //动态规划转移函数，只考虑上一步内容
            Max = max(f[i],Max);                    //输出结果
        }
        return Max;
    }
};

//动态规划--优化
//核心转移函数只用到了f[i]和f[i-1]，是一对一的数据转移，因此只需要采用一个int值进行转移即可无需向量
class Solution {
public:
    int maxSubArray(vector<int>& nums) 
    {
        
        int r = nums.size();
        int Max = INT_MIN;
    
        // vector<int> f(r,0);
        // f[0] = nums[0];
        int f(nums[0]);
        Max = f;                                 
        for(int i = 1;i < r;i++){                   
            f = max(f+nums[i],nums[i]);     //只改变了整数f  
            Max = max(f,Max);   
        }                 
        return Max;
    }
};

//贪心算法
class Solution {
public:
    int maxSubArray(vector<int>& nums) 
    {
    /*
     * 	题目要求我们只求最优解，即获取最大的值，那么很容易会考虑到贪心算法
     * 	我们需要考虑到贪心抛弃，当你的tmp值加到负值(为0其实也可以抛弃，因为没有用处)的时候，那么前面的子串和后面的字符组合只会造成负面影响(贪心负影响,通俗的说就是前面子串和后面组合还不如后面本身大)，
     * 	因此，我们贪心地舍弃掉前面的子串，重新建立子串找最大值
     * */
        int n = nums.size();
    	int Max = nums[0];
    	//保存临时值
    	int tmp = 0;
    	for(int i = 0;i < n;i++) {                  //历遍.0-n
    		tmp += nums[i];                         //临时值依次与元素相加
    		if(tmp > Max) Max = tmp;                //判断当前值与数组最大序数和大小，当大于时重新赋值Max
    		if(tmp < 0) tmp = 0;                    //当求和的临时值小于0，则归零重新开始
    	}                                           //在贪心法中，当求序数和时，当前和为负数，对和最大无任何帮助，应舍弃
    	return Max;
    }
};