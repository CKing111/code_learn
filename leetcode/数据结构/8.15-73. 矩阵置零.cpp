/*
����һ��?m x n �ľ������һ��Ԫ��Ϊ 0 �����������к��е�����Ԫ�ض���Ϊ 0 ����ʹ�� ԭ�� �㷨��

���ף�
һ��ֱ�۵Ľ��������ʹ�� ?O(mn)?�Ķ���ռ䣬���Ⲣ����һ���õĽ��������
һ���򵥵ĸĽ�������ʹ�� O(m?+?n) �Ķ���ռ䣬������Ȼ������õĽ��������
�������һ����ʹ�ó����ռ�Ľ��������
?
ʾ�� 1��
���룺matrix = [[1,1,1],[1,0,1],[1,1,1]]
�����[[1,0,1],[0,0,0],[1,0,1]]

ʾ�� 2��
���룺matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
�����[[0,0,0,0],[0,4,5,0],[0,3,1,0]]
?

��ʾ��
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

//�������
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        vector<int> row(m), col(n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!matrix[i][j]) {                //�ж�i��j�����Ƿ�Ϊ0
                    row[i] = col[j] = true;         //�趨��i�к͵�j��Ϊtrue
                }
            }
        }
        for (int i = 0; i < m; i++) {           
            for (int j = 0; j < n; j++) {
                if (row[i] || col[j]) {             //��������Ԫ�أ�ʹ����Ϊtrue��Ԫ�ظ�ֵ0
                    matrix[i][j] = 0;
                }
            }
        }
    }
};
