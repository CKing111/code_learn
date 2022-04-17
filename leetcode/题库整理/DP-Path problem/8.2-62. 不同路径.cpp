/*
https://leetcode-cn.com/problems/unique-paths/
这是 LeetCode 上的「62. 不同路径」，难度为 Medium。
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

实例1
输入：m = 3, n = 7
输出：28

实例2
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下

示例3
输入：m = 7, n = 3
输出：28

示例4
输入：m = 3, n = 3
输出：6

提示：

1 <= m, n <= 100
题目数据保证答案小于等于 2 *
*/
#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 0));       //声明dp为路径信息的二维向量
        for (int i = 0; i < m; i++) dp[i][0] = 1;           //设置j=0时为边界，只有一条路径；
        for (int j = 0; j < n; j++) dp[0][j] = 1;           //同上
        for (int i = 1; i < m; i++) {                       //历遍行
            for (int j = 1; j < n; j++) {                   //历遍列
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];     //动态规划的转移方程
                                                            //dp[i][j]代表从[0][0]到达[i][j]的路径数
                                                            //当不为边境时，最后一步只接受从上和下两个方向到达终点
            }
        }
        return dp[m - 1][n - 1];
    }
};

int main(){
    int m,n;
    cout<<"请输入路径坐标规格："<<endl;
    cin>>m;
    cin>>n;
    Solution path_nums;
    cout<<"从左上角移动到右上角一共有\n"<<path_nums.uniquePaths(m,n)<<"\n条路径。"<<endl;
    system("pause");
    return 0;
}


