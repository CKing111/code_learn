/*
https://leetcode-cn.com/problems/unique-paths-ii/

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 1 和 0 来表示。

示例 1：
输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2
解释：
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右

示例 2：
输入：obstacleGrid = [[0,1],[0,0]]
输出：1


提示：
m ==?obstacleGrid.length
n ==?obstacleGrid[i].length
1 <= m, n <= 100
obstacleGrid[i][j] 为 0 或 1


*/

#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

class Solution{
    public:
         int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
            int m = obstacleGrid.size();                    //获取路径矩阵尺寸
            int n = obstacleGrid[0].size(); 
            vector<vector<int>> dp(m, vector<int>(n, 0));   //初始化声明空的dp矩阵
            for (int i = 0; i < m && obstacleGrid[i][0] == 0; i++) dp[i][0] = 1;    //声明j列为0，表目标在列的边界上，路径只有一
            for (int j = 0; j < n && obstacleGrid[0][j] == 0; j++) dp[0][j] = 1;    //同上，行边界
            for (int i = 1; i < m; i++) {
                for (int j = 1; j < n; j++) {
                    if (obstacleGrid[i][j] == 1) continue;                          //==1表目标为障碍，跳出
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];                         //动态规划变换公式
                                                                                    //要求的目标值只与上左两格目标有关
                }
            }
            return dp[m - 1][n - 1];
    };


    //输出二维矩阵


        void print(vector<vector<int>>arr , int row,int col){
            int i = 0;
            int j = 0;
            for (i = 0;i<row;i++){
                for (j=0;j<col;j++){
                    cout<<arr[i][j]<<" "<<endl;
                }
            }
        }
};



class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
            int m = obstacleGrid.size();
            int n = obstacleGrid[0].size();

            vector<vector<int>> dp(m,vector<int>(n,0));

            for(int i = 0; i < m && obstacleGrid[i][0] == 0; i++) dp[i][0]=1;
            for(int j = 0; j < n && obstacleGrid[0][j] == 0; j++) dp[0][j]=1;
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n;j++){
                    if(obstacleGrid[i][j]==1) continue;
                    dp[i][j]=dp[i-1][j] + dp[i][j-1];
                }
            }
            return dp[m-1][n-1];
    };
};