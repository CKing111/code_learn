/*
给定一个?m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。

进阶：
一个直观的解决方案是使用 ?O(mn)?的额外空间，但这并不是一个好的解决方案。
一个简单的改进方案是使用 O(m?+?n) 的额外空间，但这仍然不是最好的解决方案。
你能想出一个仅使用常量空间的解决方案吗？
?
示例 1：
输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]

示例 2：
输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]
?

提示：
m == matrix.length
n == matrix[0].length
1 <= m, n <= 200
-231 <= matrix[i][j] <= 231 - 1
*/
#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();

        vector<vector<int>> result(m,vector<int>(n));

        for(int i = 0;i<m;i++){
            for(int j = 0;j<n;j++){
                if (matrix[i][j] == 0)
                    // int ans_m = i;
                    // int ans_n = j;
                    for(int x = 0;x<n&&x!=j;x++){
                        matrix[i][x] = matrix[i][j];
                    }
                    for(int y = 0;y<m&&y!=i;y++){
                        matrix[y][j] = matrix[i][j];
                    }
                result[i][j] = matrix[i][j];
            }
        }
        // return result;
    }
};

//标记数组
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        vector<int> row(m), col(n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!matrix[i][j]) {                //判断i，j数组是否为0
                    row[i] = col[j] = true;         //设定第i行和第j列为true
                }
            }
        }
        for (int i = 0; i < m; i++) {           
            for (int j = 0; j < n; j++) {
                if (row[i] || col[j]) {             //遍历所有元素，使行列为true的元素赋值0
                    matrix[i][j] = 0;
                }
            }
        }
    }
};
