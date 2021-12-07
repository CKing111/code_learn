/*
https://leetcode-cn.com/problems/minimum-path-sum/

给定一个包含非负整数的 m?x?n?网格?grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
说明：每次只能向下或者向右移动一步。

?
示例 1：
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。

示例 2：
输入：grid = [[1,2,3],[4,5,6]]
输出：12
?

提示：
m == grid.length
n == grid[i].length
1 <= m, n <= 200
0 <= grid[i][j] <= 100
*/


#include<iostream>
#include<vector>
using namespace std;


class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();                            //声明长宽尺寸                
        int n = grid[0].size();


        // 状态定义：dp[i][j] 表示从 [0,0] 到 [i,j] 的最小路径和
        vector<vector<int>> dp(m,vector<int>(n));       //声明二维向量，用来存储返回值

        if (grid.size()==0 || grid[0].size()==0)        //判断输入是否有效
            return 0;
        dp[0][0] = grid[0][0];                          //初始化原点数据
        for (int i=1;i<m;i++) dp[i][0]=grid[i][0] + dp[i-1][0];         //单列路径
        for (int j=1;j<n;j++) dp[0][j]=grid[0][j] + dp[0][j-1];         //单行路径
        for (int i=1;i<m;i++){                                          //历遍行列
            for(int j=1;j<n;j++){
                dp[i][j]=grid[i][j] + min(dp[i][j-1],dp[i-1][j]);       //DP的传递矩阵
            }                                                           //目标点返回值，只与上一步路径最小值与本身值相加
        }
        return dp[m-1][n-1];
    }
};

//空间压缩
class Solution {
public:
    int minPathSum4(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();

        // 状态定义：dp[i] 表示从 (0, 0) 到达第 i - 1 行的最短路径值
        vector<int> dp(n);

        // 状态初始化
        dp[0] = grid[0][0];

        // 状态转移
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j != 0) { //第一行
                    dp[j] = grid[i][j] + dp[j - 1];
                } else if (i != 0 && j == 0) { // 第一列
                    dp[j] = grid[i][j] + dp[j];
                } else if (i != 0 && j != 0) {
                    dp[j] = grid[i][j] + min(dp[j], dp[j - 1]);
                }
            }
        }

        // 返回结果
        return dp[n - 1];
    }
}

//空间压缩+优化
class Solution {
public:
// 动态规划：从起始点到终点 + 使用输入数组作为状态数组
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();

        // 状态转移
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j != 0) { //第一行
                    grid[i][j] = grid[i][j] + grid[i][j - 1];
                } else if (i != 0 && j == 0) { // 第一列
                    grid[i][j] = grid[i][j] + grid[i - 1][j];
                } else if (i != 0 && j != 0) {
                    grid[i][j] = grid[i][j] + min(grid[i - 1][j], grid[i][j - 1]);
                }
            }
        }

        // 返回结果
        return grid[m - 1][n - 1];
    }
}

int main(){
    Solution A;
    vector<vector<int>> vec;
    // int arr[] ={1,3,1,1,5,1,4,2,1};
    vec[0][0] = 1;
    vec[0][1] = 3;
    vec[0][2] = 1;
    vec[1][0] = 1;
    vec[1][1] = 5;
    vec[1][2] = 1;
    vec[2][0] = 4;
    vec[2][1] = 2;
    vec[2][2] = 1;
    // vector<vector<int>> vec[1](&arr[1],&arr[3]);
    int h;
    h = A.minPathSum(vec);
    cout<<"最短路径为："<<h<<endl;
    return 0;

}