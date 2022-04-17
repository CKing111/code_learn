
/*
给你一个 n x n 的 方形 整数数组?matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。

下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素。在下一行选择的元素和当前行所选元素最多相隔一列
（即位于正下方或者沿对角线向左或者向右的第一个元素）。具体来说，位置 (row, col) 的下一个元素应当是 (row + 1, col - 1)、(row + 1, col) 
或者 (row + 1, col + 1) 。                                                                                                                                                                                                                                                                                                                                                                                                                                                      

示例 1：
输入：matrix = [[2,1,3],[6,5,4],[7,8,9]]
输出：13
解释：下面是两条和最小的下降路径，用加粗标注：
[[2,1,3],      [[2,1,3],
 [6,5,4],       [6,5,4],
 [7,8,9]]       [7,8,9]]

示例 2：
输入：matrix = [[-19,57],[-40,-5]]
输出：-59 
解释：下面是一条和最小的下降路径，用加粗标注：
[[-19,57],
 [-40,-5]]

示例 3：
输入：matrix = [[-48]]
输出：-48
?
提示：
n == matrix.length
n == matrix[i].length
1 <= n <= 100
-100 <= matrix[i][j] <= 100
*/


 #include<iostream>
 #include<vector>
 #include <algorithm>

 using namespace std;

 class Solution {
 public:
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int m = matrix.size()+1;
        int n = matrix[0].size();
        int a;
        //初始化
        vector<vector<int>> dp(m,vector<int>(n));
        for(int i = 0;i < n;i++) dp[0][i] = matrix[0][i];

        for(int i = m-1;i >= 0;i--) {
            dp[i][0] = min(dp[i-1][0],dp[i-1][1]) + matrix[i][0];
            dp[i][n] = min(dp[i-1][m],dp[i-1][n-1]) + matrix[i][n];

            for(int j = 1;j < n;j++){
                a = min(dp[i-1][j],dp[i-1][j-1]);
                dp[i][j] = min(a,dp[i-1][j+1]) + matrix[i][j];
                // dp[j] = min(dp[j],dp[j-1],dp[j+1]) + matrix[j];

            }
        }
        // return *min_element(dp[1][n-1].begin(),dp[1][n-1].end());
        int res = INT_MAX;
        for(int i = 0;i < n;i++) res = min(res,dp[m-1][i]);
        return res;
    }
};  

//自上而下、填充
//优秀

class Solution {
public:
	int minFallingPathSum(vector<vector<int>>& matrix) 
	{
		if (matrix.empty()) return 0;
		int r = matrix.size();                                      //读取尺寸，从1开始
		vector<vector<int>> dp(r + 1, vector<int>(r + 2,0));        //声明，填充后的
                                                                    //最上面填充一行，左右各填充一列
		//套壳处理---两边均为最大值
		for (int i = 0; i < r+1; i++)
		{
			dp[i][0] = INT_MAX;                     //矩阵左边添加1列
			dp[i][r+1] = INT_MAX;                   //矩阵右边添加一列
		}
		for (int i = 1; i <r+1; i++)
		{
			for (int j = 1; j < r+1; j++)
			{
				dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i - 1][j + 1])) + matrix[i-1][j-1];   //自上而下，min只能比较两个int
			}                                                                               //自上而下，取原行值
		}
		int Min = INT_MAX;
		for (int i = 0; i < r + 2; i++)
			Min = min(Min, dp[r][i]);       //历遍最后一行取最小
		return Min;
	}
};  








