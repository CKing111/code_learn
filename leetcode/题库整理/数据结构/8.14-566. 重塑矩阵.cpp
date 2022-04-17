/*
�� MATLAB �У���һ���ǳ����õĺ��� reshape �������Խ�һ��?m x n ��������Ϊ��һ����С��ͬ��r x c�����¾��󣬵�������ԭʼ���ݡ�
����һ���ɶ�ά���� mat ��ʾ��?m x n �����Լ����������� r �� c ���ֱ��ʾ��Ҫ���ع��ľ����������������
�ع���ľ�����Ҫ��ԭʼ���������Ԫ������ͬ�� �б���˳�� ��䡣
������и��������� reshape �����ǿ����Һ���ģ�������µ����ܾ��󣻷������ԭʼ����

ʾ�� 1��
���룺mat = [[1,2],[3,4]], r = 1, c = 4
�����[[1,2,3,4]]

ʾ�� 2��
���룺mat = [[1,2],[3,4]], r = 2, c = 4
�����[[1,2],[3,4]]
*/

#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> matrixReshape(vector<vector<int>>& mat, int r, int c) {
        int m = mat.size();
        int n = mat[0].size();

        if(m*n != r*c) return mat;                      //��ʼ���������任ǰ����Ԫ������ͬ

        vector<vector<int>> result(r,vector<int>(c));   //��������Ķ�ά����

        for(int x = 0;x < m*n;++x){                     //������Ԫ��
            result[x/c][x%c] = mat[x/n][x%n];           //��ͬ�ߴ����Ԫ�ص�ӳ�乫ʽ
        }                                               //һάԪ������������Ԫ�ظ����������У�����Ԫ����ȡ���������λ��
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