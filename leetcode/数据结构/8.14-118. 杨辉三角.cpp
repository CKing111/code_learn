/*
给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。
在「杨辉三角」中，每个数是它左上方和右上方的数的和。
               1
        1           1
    1           2           1
1       3           3           1
示例 1:
输入: numRows = 5
输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]

示例?2:
输入: numRows = 1
输出: [[1]]
?
提示:
1 <= numRows <= 30
*/

#include<iostream>
#include<vector>

using namespace std;


//DP

class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        //声明输出向量
        vector<vector<int>> res(numRows);
        
        //循环构造三角
        for(int i = 0;i < numRows;++i){
            //声明规则，行元素数量递加，行首和行末为1；
            res[i].resize(i+1);                             //行元素量//第i行
            res[i][0] = res[i][i] = 1;                      //行首行末
            for(int j = 1;j<i;++j){                         //遍历  //j=1和j<i表面略过了首尾元素   
                    //遍历每行，移除首尾默认元素
                res[i][j] = res[i-1][j] + res[i-1][j-1];    //转移函数
            }
        }
        return res;
    }
};