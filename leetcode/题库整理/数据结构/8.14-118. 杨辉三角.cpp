/*
����һ���Ǹ����� numRows�����ɡ�������ǡ���ǰ numRows �С�
�ڡ�������ǡ��У�ÿ�����������Ϸ������Ϸ������ĺ͡�
               1
        1           1
    1           2           1
1       3           3           1
ʾ�� 1:
����: numRows = 5
���: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]

ʾ��?2:
����: numRows = 1
���: [[1]]
?
��ʾ:
1 <= numRows <= 30
*/

#include<iostream>
#include<vector>

using namespace std;


//DP

class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        //�����������
        vector<vector<int>> res(numRows);
        
        //ѭ����������
        for(int i = 0;i < numRows;++i){
            //����������Ԫ�������ݼӣ����׺���ĩΪ1��
            res[i].resize(i+1);                             //��Ԫ����//��i��
            res[i][0] = res[i][i] = 1;                      //������ĩ
            for(int j = 1;j<i;++j){                         //����  //j=1��j<i�����Թ�����βԪ��   
                    //����ÿ�У��Ƴ���βĬ��Ԫ��
                res[i][j] = res[i-1][j] + res[i-1][j-1];    //ת�ƺ���
            }
        }
        return res;
    }
};