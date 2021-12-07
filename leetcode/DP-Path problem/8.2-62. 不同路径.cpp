/*
https://leetcode-cn.com/problems/unique-paths/
���� LeetCode �ϵġ�62. ��ͬ·�������Ѷ�Ϊ Medium��
һ��������λ��һ�� m x n ��������Ͻ� ����ʼ������ͼ�б��Ϊ ��Start�� ����
������ÿ��ֻ�����»��������ƶ�һ������������ͼ�ﵽ��������½ǣ�����ͼ�б��Ϊ ��Finish�� ����

ʵ��1
���룺m = 3, n = 7
�����28

ʵ��2
���룺m = 3, n = 2
�����3
���ͣ�
�����Ͻǿ�ʼ���ܹ��� 3 ��·�����Ե������½ǡ�
1. ���� -> ���� -> ����
2. ���� -> ���� -> ����
3. ���� -> ���� -> ����

ʾ��3
���룺m = 7, n = 3
�����28

ʾ��4
���룺m = 3, n = 3
�����6

��ʾ��

1 <= m, n <= 100
��Ŀ���ݱ�֤��С�ڵ��� 2 *
*/
#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 0));       //����dpΪ·����Ϣ�Ķ�ά����
        for (int i = 0; i < m; i++) dp[i][0] = 1;           //����j=0ʱΪ�߽磬ֻ��һ��·����
        for (int j = 0; j < n; j++) dp[0][j] = 1;           //ͬ��
        for (int i = 1; i < m; i++) {                       //������
            for (int j = 1; j < n; j++) {                   //������
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];     //��̬�滮��ת�Ʒ���
                                                            //dp[i][j]�����[0][0]����[i][j]��·����
                                                            //����Ϊ�߾�ʱ�����һ��ֻ���ܴ��Ϻ����������򵽴��յ�
            }
        }
        return dp[m - 1][n - 1];
    }
};

int main(){
    int m,n;
    cout<<"������·��������"<<endl;
    cin>>m;
    cin>>n;
    Solution path_nums;
    cout<<"�����Ͻ��ƶ������Ͻ�һ����\n"<<path_nums.uniquePaths(m,n)<<"\n��·����"<<endl;
    system("pause");
    return 0;
}


