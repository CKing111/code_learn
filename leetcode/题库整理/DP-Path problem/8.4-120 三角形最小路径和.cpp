/*
https://leetcode-cn.com/problems/triangle/

120��

����һ�������� triangle ���ҳ��Զ����µ���С·���͡�
ÿһ��ֻ���ƶ�����һ�������ڵĽ���ϡ����ڵĽ�� ������ָ���� �±� �� ��һ�����±� ��ͬ���ߵ��� 
��һ�����±� + 1 ��������㡣Ҳ����˵�������λ�ڵ�ǰ�е��±� i ����ô��һ�������ƶ�����һ�е��±� i �� i + 1 ��

ʾ�� 1��
���룺triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
�����11
���ͣ��������ͼ��ʾ��
   2
  3 4
 6 5 7
4 1 8 3
�Զ����µ���С·����Ϊ?11������2?+?3?+?5?+?1?= 11����

ʾ�� 2��
���룺triangle = [[-10]]
�����-10
?
��ʾ��
1 <= triangle.length <= 200
triangle[0].length == 1
triangle[i].length == triangle[i - 1].length + 1
-104 <= triangle[i][j] <= 104
?
���ף�
�����ֻʹ�� O(n)?�Ķ���ռ䣨n Ϊ�����ε�����������������������
*/


#include<iostream>
#include<vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int m = triangle.size();                                //�����ߴ磬�гߴ�����гߴ�
        // int n = triangle[0].size();
        
        vector<vector<int>> dp(m,vector<int>(m));               //������ά����
                                                                //dp[i][j]������ijλ�õ���С·��
        dp[0][0] = triangle[0][0];                              //��ʼ��[0][0]λ��

        for(int i = 1;i < m; i++) {                             //�������гߴ磬��1��ʼ
        dp[i][0] = dp[i-1][0] + triangle[i][0];                 //�������������·
        dp[i][i] = dp[i-1][i-1] + triangle[i][i];               //���������Ҷ���·
            for(int j = 1;j<i ;j++){                            //������ѭ����������ά��
                dp[i][j] = min(dp[i-1][j-1] , dp[i-1][j]) + triangle[i][j];     //��̬�滮��ת�ƺ���
            }
        }
        return  *min_element(dp[m - 1].begin(), dp[m - 1].end());//������������е���Сֵ��
    }
};
//�ռ��Ż�
//���¶���
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {

        //�ܾ����dp��Ŀ
        int row = triangle.size();   //������,��1��ʼ��ȡ��

        vector<int> dp(row + 1,0);    //���ݶ����У�������ȷ����Ҫ���ĸ����ռ�  ���ｫ�����Ԫ�ض���ʼ��Ϊ0

        for (int i = row - 1; i >= 0; i--)  //�����һ�п�ʼ���һ����  �����µ���
        {       //-1����ȡ0��
            for (int j = 0; j <triangle[i].size(); j++)    //�ӵ�һ�������һ���ߣ� ������
            {                       //����������[]
                dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j]; //����ѡ����С��Ԫ�� Ȼ���ټ���Ҫ�����Ԫ��
            }                       //�����¶��ϣ������º���       //triangle�Ƕ�άֵ
        }
        return dp[0];
    }
};

//���
//���¶���
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        vector<int> dp(triangle.back());
        for(int i = triangle.size() - 2; i >= 0; i --)
            for(int j = 0; j <= i; j ++)
                dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j];
        return dp[0];
    }
};

