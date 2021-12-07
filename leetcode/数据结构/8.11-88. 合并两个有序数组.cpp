/*
给你两个有序整数数组?nums1 和 nums2，请你将 nums2 合并到?nums1?中，
使 nums1 成为一个有序数组。初始化?nums1 和 nums2 的元素数量分别为?m 和 n 。
你可以假设?nums1 的空间大小等于?m + n，这样它就有足够的空间保存来自 nums2 的元素。

示例 1：
输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]

示例 2：
输入：nums1 = [1], m = 1, nums2 = [], n = 0
输出：[1]
?
提示：
nums1.length == m + n
nums2.length == n
0 <= m, n <= 200
1 <= m + n <= 200
-109 <= nums1[i], nums2[i] <= 109
*/

#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

//暴力枚举
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        for (int i = 0;i<n;i++){
            nums1[m+i] = nums2[i];
        }
        return sort(nums1.begin(),nums1.end());
    }
};


//从后向前对比写入
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int k = m + n - 1;                                      //声明合并后总大小
        while (m > 0 && n > 0) {                                //条件判断，非0时继续
            if (nums1[m - 1] > nums2[n - 1]) {                  //对比1和2的倒数元素大小，将大的写入新组合的末尾
                nums1[k] = nums1[m - 1];                        
                m --;
            } else {
                nums1[k] = nums2[n - 1];                        //同上，并向左移动写入值
                n --;
            }
            k --;                                               //从后往前依次写入新数组
        }
        for (int i = 0; i < n; ++i) {                           //当阶数while循环时，有一个数组已经写入完毕
            nums1[i] = nums2[i];                                //由于时对比后写入，因此余下的一定是小于写入且是有序数组，一次写入余下位置
        }
    }
};
