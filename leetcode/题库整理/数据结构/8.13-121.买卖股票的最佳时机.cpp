/*
����һ������ prices �����ĵ�?i ��Ԫ��?prices[i] ��ʾһ֧������Ʊ�� i ��ļ۸�
��ֻ��ѡ�� ĳһ�� ������ֻ��Ʊ����ѡ���� δ����ĳһ����ͬ������ �����ù�Ʊ�����һ���㷨�����������ܻ�ȡ���������
��������Դ���ʽ����л�ȡ�������������㲻�ܻ�ȡ�κ����󣬷��� 0 ��
?
ʾ�� 1��
���룺[7,1,5,3,6,4]
�����5
���ͣ��ڵ� 2 �죨��Ʊ�۸� = 1����ʱ�����룬�ڵ� 5 �죨��Ʊ�۸� = 6����ʱ��������������� = 6-1 = 5 ��
     ע���������� 7-1 = 6, ��Ϊ�����۸���Ҫ��������۸�ͬʱ���㲻��������ǰ������Ʊ��

ʾ�� 2��
���룺prices = [7,6,4,3,1]
�����0
���ͣ������������, û�н������, �����������Ϊ 0��

��ʾ��
1 <= prices.length <= 105
0 <= prices[i] <= 104
*/

#include<iostream>
#include<vector>

using namespace std;

//����ö��
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = (int)prices.size(), ans = 0;    //������ʼ��
        for (int i = 0; i < n; ++i){
            for (int j = i + 1; j < n; ++j) {
                ans = max(ans, prices[j] - prices[i]);  //�Ƚϵ�ǰ����Ĳ�ֵ���ʼ���Ĵ�С������
            }
        }
        return ans;
    }
};


//��̬�滮
//�������飬��¼�����ڵ���͵�
//������͵����룬Ȼ������֮������ֵ������������

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int inf = 1e9;                                          
        int minprice = inf, maxprofit = 0;                      //������СԪ��ֵ���������ֵ
        for (int price: prices) {                               //����prices����
            maxprofit = max(maxprofit, price - minprice);       //��¼�������������ֵ
            minprice = min(price, minprice);                    //��¼��������е���СֵԪ��
        }
        return maxprofit;
    }
};

