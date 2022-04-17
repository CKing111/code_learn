/*
https://leetcode-cn.com/problems/unique-paths-ii/

һ��������λ��һ�� m x n ��������Ͻ� ����ʼ������ͼ�б��Ϊ��Start�� ����
������ÿ��ֻ�����»��������ƶ�һ������������ͼ�ﵽ��������½ǣ�����ͼ�б��Ϊ��Finish������
���ڿ������������ϰ����ô�����Ͻǵ����½ǽ����ж�������ͬ��·����

�����е��ϰ���Ϳ�λ�÷ֱ��� 1 �� 0 ����ʾ��

ʾ�� 1��
���룺obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
�����2
���ͣ�
3x3 ��������м���һ���ϰ��
�����Ͻǵ����½�һ���� 2 ����ͬ��·����
1. ���� -> ���� -> ���� -> ����
2. ���� -> ���� -> ���� -> ����

ʾ�� 2��
���룺obstacleGrid = [[0,1],[0,0]]
�����1


��ʾ��
m ==?obstacleGrid.length
n ==?obstacleGrid[i].length
1 <= m, n <= 100
obstacleGrid[i][j] Ϊ 0 �� 1


*/

#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

class Solution{
    public:
         int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
            int m = obstacleGrid.size();                    //��ȡ·������ߴ�
            int n = obstacleGrid[0].size(); 
            vector<vector<int>> dp(m, vector<int>(n, 0));   //��ʼ�������յ�dp����
            for (int i = 0; i < m && obstacleGrid[i][0] == 0; i++) dp[i][0] = 1;    //����j��Ϊ0����Ŀ�����еı߽��ϣ�·��ֻ��һ
            for (int j = 0; j < n && obstacleGrid[0][j] == 0; j++) dp[0][j] = 1;    //ͬ�ϣ��б߽�
            for (int i = 1; i < m; i++) {
                for (int j = 1; j < n; j++) {
                    if (obstacleGrid[i][j] == 1) continue;                          //==1��Ŀ��Ϊ�ϰ�������
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];                         //��̬�滮�任��ʽ
                                                                                    //Ҫ���Ŀ��ֵֻ����������Ŀ���й�
                }
            }
            return dp[m - 1][n - 1];
    };


    //�����ά����


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