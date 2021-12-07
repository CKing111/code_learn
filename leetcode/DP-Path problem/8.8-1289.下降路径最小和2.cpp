/*
����һ����������?arr?�����塸����ƫ���½�·����Ϊ����?arr �����е�ÿһ��ѡ��һ�����֣��Ұ�˳��ѡ�����������У��������ֲ���ԭ�����ͬһ�С�

���㷵�ط���ƫ���½�·�����ֺ͵���Сֵ��

?

ʾ�� 1��
���룺arr = [[1,2,3],[4,5,6],[7,8,9]]
�����13
���ͣ�
���з���ƫ���½�·��������
[1,5,9], [1,5,7], [1,6,7], [1,6,8],
[2,4,8], [2,4,9], [2,6,7], [2,6,8],
[3,4,8], [3,4,9], [3,5,7], [3,5,9]
�½�·�������ֺ���С����?[1,5,7] �����Դ���?13 ��
?

��ʾ��
1 <= arr.length == arr[i].length <= 200
-99 <= arr[i][j] <= 99
*/

#include<iostream>
#include<vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minFallingPathSum(vector<vector<int>>& arr) {
            if (arr.empty()) return 0;
            int r = arr.size();     //1-arr.size()

            vector<vector<int>> dp(r+1,vector<int>(r+2,0));
            // vector<int>& dp(r+1,0);

            for(int i = 0;i < r+1;i++){
                dp[i][0] = INT_MAX;
                dp[i][r+1] = INT_MAX;
            } 
            for(int i = 1;i<r+1;i++){
                for(int j = 1;j<r+1;j++){
                    dp[i][j] = min(dp[i-1][j-1],dp[i-1][j+1]) + arr[i][j];
                }
            }
            int Min = INT_MAX;
            for(int i = 0;i<r+2;i++){
                Min = min(Min,dp[r+1][i]);
            }
            return Min;
    }
};  