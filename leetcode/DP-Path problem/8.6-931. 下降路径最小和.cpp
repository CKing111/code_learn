
/*
����һ�� n x n �� ���� ��������?matrix �������ҳ�������ͨ�� matrix ���½�·�� �� ��С�� ��

�½�·�� ���Դӵ�һ���е��κ�Ԫ�ؿ�ʼ������ÿһ����ѡ��һ��Ԫ�ء�����һ��ѡ���Ԫ�غ͵�ǰ����ѡԪ��������һ��
����λ�����·������ضԽ�������������ҵĵ�һ��Ԫ�أ���������˵��λ�� (row, col) ����һ��Ԫ��Ӧ���� (row + 1, col - 1)��(row + 1, col) 
���� (row + 1, col + 1) ��                                                                                                                                                                                                                                                                                                                                                                                                                                                      

ʾ�� 1��
���룺matrix = [[2,1,3],[6,5,4],[7,8,9]]
�����13
���ͣ���������������С���½�·�����üӴֱ�ע��
[[2,1,3],      [[2,1,3],
 [6,5,4],       [6,5,4],
 [7,8,9]]       [7,8,9]]

ʾ�� 2��
���룺matrix = [[-19,57],[-40,-5]]
�����-59 
���ͣ�������һ������С���½�·�����üӴֱ�ע��
[[-19,57],
 [-40,-5]]

ʾ�� 3��
���룺matrix = [[-48]]
�����-48
?
��ʾ��
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
        //��ʼ��
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

//���϶��¡����
//����

class Solution {
public:
	int minFallingPathSum(vector<vector<int>>& matrix) 
	{
		if (matrix.empty()) return 0;
		int r = matrix.size();                                      //��ȡ�ߴ磬��1��ʼ
		vector<vector<int>> dp(r + 1, vector<int>(r + 2,0));        //�����������
                                                                    //���������һ�У����Ҹ����һ��
		//�׿Ǵ���---���߾�Ϊ���ֵ
		for (int i = 0; i < r+1; i++)
		{
			dp[i][0] = INT_MAX;                     //����������1��
			dp[i][r+1] = INT_MAX;                   //�����ұ����һ��
		}
		for (int i = 1; i <r+1; i++)
		{
			for (int j = 1; j < r+1; j++)
			{
				dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i - 1][j + 1])) + matrix[i-1][j-1];   //���϶��£�minֻ�ܱȽ�����int
			}                                                                               //���϶��£�ȡԭ��ֵ
		}
		int Min = INT_MAX;
		for (int i = 0; i < r + 2; i++)
			Min = min(Min, dp[r][i]);       //�������һ��ȡ��С
		return Min;
	}
};  








