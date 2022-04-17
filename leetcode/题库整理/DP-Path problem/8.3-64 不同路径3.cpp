/*
https://leetcode-cn.com/problems/minimum-path-sum/

����һ�������Ǹ������� m?x?n?����?grid �����ҳ�һ�������Ͻǵ����½ǵ�·����ʹ��·���ϵ������ܺ�Ϊ��С��
˵����ÿ��ֻ�����»��������ƶ�һ����

?
ʾ�� 1��
���룺grid = [[1,3,1],[1,5,1],[4,2,1]]
�����7
���ͣ���Ϊ·�� 1��3��1��1��1 ���ܺ���С��

ʾ�� 2��
���룺grid = [[1,2,3],[4,5,6]]
�����12
?

��ʾ��
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
        int m = grid.size();                            //��������ߴ�                
        int n = grid[0].size();


        // ״̬���壺dp[i][j] ��ʾ�� [0,0] �� [i,j] ����С·����
        vector<vector<int>> dp(m,vector<int>(n));       //������ά�����������洢����ֵ

        if (grid.size()==0 || grid[0].size()==0)        //�ж������Ƿ���Ч
            return 0;
        dp[0][0] = grid[0][0];                          //��ʼ��ԭ������
        for (int i=1;i<m;i++) dp[i][0]=grid[i][0] + dp[i-1][0];         //����·��
        for (int j=1;j<n;j++) dp[0][j]=grid[0][j] + dp[0][j-1];         //����·��
        for (int i=1;i<m;i++){                                          //��������
            for(int j=1;j<n;j++){
                dp[i][j]=grid[i][j] + min(dp[i][j-1],dp[i-1][j]);       //DP�Ĵ��ݾ���
            }                                                           //Ŀ��㷵��ֵ��ֻ����һ��·����Сֵ�뱾��ֵ���
        }
        return dp[m-1][n-1];
    }
};

//�ռ�ѹ��
class Solution {
public:
    int minPathSum4(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();

        // ״̬���壺dp[i] ��ʾ�� (0, 0) ����� i - 1 �е����·��ֵ
        vector<int> dp(n);

        // ״̬��ʼ��
        dp[0] = grid[0][0];

        // ״̬ת��
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j != 0) { //��һ��
                    dp[j] = grid[i][j] + dp[j - 1];
                } else if (i != 0 && j == 0) { // ��һ��
                    dp[j] = grid[i][j] + dp[j];
                } else if (i != 0 && j != 0) {
                    dp[j] = grid[i][j] + min(dp[j], dp[j - 1]);
                }
            }
        }

        // ���ؽ��
        return dp[n - 1];
    }
}

//�ռ�ѹ��+�Ż�
class Solution {
public:
// ��̬�滮������ʼ�㵽�յ� + ʹ������������Ϊ״̬����
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();

        // ״̬ת��
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j != 0) { //��һ��
                    grid[i][j] = grid[i][j] + grid[i][j - 1];
                } else if (i != 0 && j == 0) { // ��һ��
                    grid[i][j] = grid[i][j] + grid[i - 1][j];
                } else if (i != 0 && j != 0) {
                    grid[i][j] = grid[i][j] + min(grid[i - 1][j], grid[i][j - 1]);
                }
            }
        }

        // ���ؽ��
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
    cout<<"���·��Ϊ��"<<h<<endl;
    return 0;

}