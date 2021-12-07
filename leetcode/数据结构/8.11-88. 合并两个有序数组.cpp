/*
��������������������?nums1 �� nums2�����㽫 nums2 �ϲ���?nums1?�У�
ʹ nums1 ��Ϊһ���������顣��ʼ��?nums1 �� nums2 ��Ԫ�������ֱ�Ϊ?m �� n ��
����Լ���?nums1 �Ŀռ��С����?m + n�������������㹻�Ŀռ䱣������ nums2 ��Ԫ�ء�

ʾ�� 1��
���룺nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
�����[1,2,2,3,5,6]

ʾ�� 2��
���룺nums1 = [1], m = 1, nums2 = [], n = 0
�����[1]
?
��ʾ��
nums1.length == m + n
nums2.length == n
0 <= m, n <= 200
1 <= m + n <= 200
-109 <= nums1[i], nums2[i] <= 109
*/

#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

//����ö��
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        for (int i = 0;i<n;i++){
            nums1[m+i] = nums2[i];
        }
        return sort(nums1.begin(),nums1.end());
    }
};


//�Ӻ���ǰ�Ա�д��
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int k = m + n - 1;                                      //�����ϲ����ܴ�С
        while (m > 0 && n > 0) {                                //�����жϣ���0ʱ����
            if (nums1[m - 1] > nums2[n - 1]) {                  //�Ա�1��2�ĵ���Ԫ�ش�С�������д������ϵ�ĩβ
                nums1[k] = nums1[m - 1];                        
                m --;
            } else {
                nums1[k] = nums2[n - 1];                        //ͬ�ϣ��������ƶ�д��ֵ
                n --;
            }
            k --;                                               //�Ӻ���ǰ����д��������
        }
        for (int i = 0; i < n; ++i) {                           //������whileѭ��ʱ����һ�������Ѿ�д�����
            nums1[i] = nums2[i];                                //����ʱ�ԱȺ�д�룬������µ�һ����С��д�������������飬һ��д������λ��
        }
    }
};
