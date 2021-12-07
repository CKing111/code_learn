/*
在 MATLAB 中，有一个非常有用的函数 reshape ，它可以将一个?m x n 矩阵重塑为另一个大小不同（r x c）的新矩阵，但保留其原始数据。
给你一个由二维数组 mat 表示的?m x n 矩阵，以及两个正整数 r 和 c ，分别表示想要的重构的矩阵的行数和列数。
重构后的矩阵需要将原始矩阵的所有元素以相同的 行遍历顺序 填充。
如果具有给定参数的 reshape 操作是可行且合理的，则输出新的重塑矩阵；否则，输出原始矩阵。

示例 1：
输入：mat = [[1,2],[3,4]], r = 1, c = 4
输出：[[1,2,3,4]]

示例 2：
输入：mat = [[1,2],[3,4]], r = 2, c = 4
输出：[[1,2],[3,4]]
*/

#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> matrixReshape(vector<vector<int>>& mat, int r, int c) {
        int m = mat.size();
        int n = mat[0].size();

        if(m*n != r*c) return mat;                      //初始条件，即变换前后总元素数相同

        vector<vector<int>> result(r,vector<int>(c));   //声明输出的二维向量

        for(int x = 0;x < m*n;++x){                     //迭代总元素
            result[x/c][x%c] = mat[x/n][x%n];           //不同尺寸矩阵元素的映射公式
        }                                               //一维元素索引除以行元素个数得所在行，除行元素数取余表所处行位置
        return result;
    }
};

class Solution {
public:
    vector<vector<int>> matrixReshape(vector<vector<int>>& mat, int r, int c) {
        int m = mat.size(), n = mat[0].size();
        if(m * n != r * c) return mat;
        vector<vector<int>> res;

        {
            for(int j = 0; j < r; j++)
            {
                vector<int> temp;
                for(int i = 0; i < c; i++)
                {
                    temp.push_back(mat[(j*c+i)/n][(j*c+i)%n]);
                }
                res.push_back(temp);
            }
        }

        return res;
    }
};