/*
给定一个整数数组，判断是否存在重复元素。
如果存在一值在数组中出现至少两次，函数返回 true 。如果数组中每个元素都不相同，则返回 false 。

 
示例 1:
输入: [1,2,3,1]
输出: true

示例 2:
输入: [1,2,3,4]
输出: false

示例 3:
输入: [1,1,1,3,3,4,3,2,4,2]
输出: true
*/

#include<iostream>
#include<vector>
#include < algorithm>

using namespace std;


//1
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        int r = nums.size();//1-n
        int num1,num2;
        for (int i = 1;i < r +1;i++){
            num1 = nums[i];
            for (int j = i+1;j < r+1 ;j++){
                num2 = nums[j];
                if (num1 == num2) return true;
            }
        }
    return false;
    }   
};


//2排序


class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        sort(nums.begin(), nums.end());                         //排序
        int n = nums.size();                                                    //1-n
        for (int i = 0; i < n - 1; i++) {                                       //0-n
            if (nums[i] == nums[i + 1]) {                                 //判断相邻元素是否一样
                return true;
            }
        }
        return false;
    }
};

//哈希表

class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_set<int> s;                                                       //初始化哈希表
        for (int x: nums) {             
            if (s.find(x) != s.end()) {                                                     
                return true;
            }
            s.insert(x);
        }
        return false;
    }
};

