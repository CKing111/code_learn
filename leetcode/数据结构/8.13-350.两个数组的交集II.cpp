/*
给定两个数组，编写一个函数来计算它们的交集。

示例 1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]

示例 2:
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]
?
说明：
输出结果中每个元素出现的次数，应与元素在两个数组中出现次数的最小值一致。
我们可以不考虑输出结果的顺序。

进阶：
如果给定的数组已经排好序呢？你将如何优化你的算法？
如果?nums1?的大小比?nums2?小很多，哪种方法更优？
如果?nums2?的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？
*/

//哈希表

#include<iostream>
#include<vector>
#include<unordered_map>
using namespace std;

class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        //判断输入两数组的大小，有限历遍小的数组
        //若小在后，则对调intersect输入中两数组的位置
        if (nums1.size() > nums2.size()) {
            return intersect(nums2, nums1);     //实际输入中的nums1变成调换后的nums2
        }
        //第一阶段
        //声明一个无序map来存储第一个数组中出现的元素和次数
        unordered_map <int, int> m;
        for (int num : nums1) {         //C++11新特性，for（定义的迭代变量:迭代范围）
            ++m[num];                   //写入map
        }
        //第二阶段
        //在历遍第二个数组，声明一个输出结果的向量
        //当第二个数组出现第一个数组的元素时，写入输出并将map中出现次数-1
        vector<int> intersection;
        for (int num : nums2) {
            if (m.count(num)) {                 //count(key):在容器中查找以 key 键的键值对的个数。
                                                //m.count(num) = true
                intersection.push_back(num);    //添加到输出中
                --m[num];                       //m中相同数组-1，因此次序-1
                if (m[num] == 0) {              //无重复-----已经通过迭代出现次数降为0
                    m.erase(num);               //删除该键对，表示存在重复并集
                }
            }
        }
        return intersection;                    //输出
    }
};

//排序+双指针
//思想：生命两个指针，来依次指向两个数组，并对比指向元素，
//将相同的元素写入输出，若不相同，则将小值指针右移一个位置再进行比对，当指针溢出则执行结束
#include <algorithm>

class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        //排序两数组，默认升序
        sort(nums1.begin(), nums1.end());
        sort(nums2.begin(), nums2.end());
        //声明结果容器，读取长度，声明两个指针
        int length1 = nums1.size(), length2 = nums2.size();
        vector<int> intersection;
        int index1 = 0, index2 = 0;
        //声明边界，超出阶数
        //对比大小，写入想等结果
        while (index1 < length1 && index2 < length2) {
            if (nums1[index1] < nums2[index2]) {
                index1++;                                   //当nums1中指针指向元素小，则右移该指针
            } else if (nums1[index1] > nums2[index2]) {
                index2++;                                   //同上
            } else {                                        //排除小于大于条件以后，
                intersection.push_back(nums1[index1]);      //将重复元素写入输出
                index1++;
                index2++;
            }
        }
        return intersection;
    }
};
