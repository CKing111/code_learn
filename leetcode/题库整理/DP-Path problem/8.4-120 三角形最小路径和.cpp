/*
https://leetcode-cn.com/problems/triangle/

120题

给定一个三角形 triangle ，找出自顶向下的最小路径和。
每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 
上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

示例 1：
输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
输出：11
解释：如下面简图所示：
   2
  3 4
 6 5 7
4 1 8 3
自顶向下的最小路径和为?11（即，2?+?3?+?5?+?1?= 11）。

示例 2：
输入：triangle = [[-10]]
输出：-10
?
提示：
1 <= triangle.length <= 200
triangle[0].length == 1
triangle[i].length == triangle[i - 1].length + 1
-104 <= triangle[i][j] <= 104
?
进阶：
你可以只使用 O(n)?的额外空间（n 为三角形的总行数）来解决这个问题吗？
*/


#include<iostream>
#include<vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int m = triangle.size();                                //声明尺寸，行尺寸等于列尺寸
        // int n = triangle[0].size();
        
        vector<vector<int>> dp(m,vector<int>(m));               //声明二维向量
                                                                //dp[i][j]代表到达ij位置的最小路径
        dp[0][0] = triangle[0][0];                              //初始化[0][0]位置

        for(int i = 1;i < m; i++) {                             //历遍行列尺寸，从1开始
        dp[i][0] = dp[i-1][0] + triangle[i][0];                 //金字塔最左侧线路
        dp[i][i] = dp[i-1][i-1] + triangle[i][i];               //金字塔最右端线路
            for(int j = 1;j<i ;j++){                            //结合外层循环，遍历二维点
                dp[i][j] = min(dp[i-1][j-1] , dp[i-1][j]) + triangle[i][j];     //动态规划的转移函数
            }
        }
        return  *min_element(dp[m - 1].begin(), dp[m - 1].end());//输出整个过程中的最小值；
    }
};
//空间优化
//自下而上
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {

        //很经典的dp题目
        int row = triangle.size();   //多少行,从1开始读取行

        vector<int> dp(row + 1,0);    //根据多少行，我们来确定需要多大的辅助空间  这里将里面的元素都初始化为0

        for (int i = row - 1; i >= 0; i--)  //从最后一行开始向第一行走  即从下到上
        {       //-1，读取0行
            for (int j = 0; j <triangle[i].size(); j++)    //从第一列向最后一列走， 从左到右
            {                       //向量索引用[]
                dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j]; //先再选择最小的元素 然后再加上要计算的元素
            }                       //↑自下而上，返回下和右       //triangle是二维值
        }
        return dp[0];
    }
};

//最佳
//自下而上
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        vector<int> dp(triangle.back());
        for(int i = triangle.size() - 2; i >= 0; i --)
            for(int j = 0; j <= i; j ++)
                dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j];
        return dp[0];
    }
};

