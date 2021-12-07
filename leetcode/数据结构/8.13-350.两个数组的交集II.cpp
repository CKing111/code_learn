/*
�����������飬��дһ���������������ǵĽ�����

ʾ�� 1��
���룺nums1 = [1,2,2,1], nums2 = [2,2]
�����[2,2]

ʾ�� 2:
���룺nums1 = [4,9,5], nums2 = [9,4,9,8,4]
�����[4,9]
?
˵����
��������ÿ��Ԫ�س��ֵĴ�����Ӧ��Ԫ�������������г��ִ�������Сֵһ�¡�
���ǿ��Բ�������������˳��

���ף�
��������������Ѿ��ź����أ��㽫����Ż�����㷨��
���?nums1?�Ĵ�С��?nums2?С�ܶ࣬���ַ������ţ�
���?nums2?��Ԫ�ش洢�ڴ����ϣ��ڴ������޵ģ������㲻��һ�μ������е�Ԫ�ص��ڴ��У������ô�죿
*/

//��ϣ��

#include<iostream>
#include<vector>
#include<unordered_map>
using namespace std;

class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        //�ж�����������Ĵ�С����������С������
        //��С�ں���Ե�intersect�������������λ��
        if (nums1.size() > nums2.size()) {
            return intersect(nums2, nums1);     //ʵ�������е�nums1��ɵ������nums2
        }
        //��һ�׶�
        //����һ������map���洢��һ�������г��ֵ�Ԫ�غʹ���
        unordered_map <int, int> m;
        for (int num : nums1) {         //C++11�����ԣ�for������ĵ�������:������Χ��
            ++m[num];                   //д��map
        }
        //�ڶ��׶�
        //������ڶ������飬����һ��������������
        //���ڶ���������ֵ�һ�������Ԫ��ʱ��д���������map�г��ִ���-1
        vector<int> intersection;
        for (int num : nums2) {
            if (m.count(num)) {                 //count(key):�������в����� key ���ļ�ֵ�Եĸ�����
                                                //m.count(num) = true
                intersection.push_back(num);    //��ӵ������
                --m[num];                       //m����ͬ����-1����˴���-1
                if (m[num] == 0) {              //���ظ�-----�Ѿ�ͨ���������ִ�����Ϊ0
                    m.erase(num);               //ɾ���ü��ԣ���ʾ�����ظ�����
                }
            }
        }
        return intersection;                    //���
    }
};

//����+˫ָ��
//˼�룺��������ָ�룬������ָ���������飬���Ա�ָ��Ԫ�أ�
//����ͬ��Ԫ��д�������������ͬ����Сֵָ������һ��λ���ٽ��бȶԣ���ָ�������ִ�н���
#include <algorithm>

class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        //���������飬Ĭ������
        sort(nums1.begin(), nums1.end());
        sort(nums2.begin(), nums2.end());
        //���������������ȡ���ȣ���������ָ��
        int length1 = nums1.size(), length2 = nums2.size();
        vector<int> intersection;
        int index1 = 0, index2 = 0;
        //�����߽磬��������
        //�Աȴ�С��д����Ƚ��
        while (index1 < length1 && index2 < length2) {
            if (nums1[index1] < nums2[index2]) {
                index1++;                                   //��nums1��ָ��ָ��Ԫ��С�������Ƹ�ָ��
            } else if (nums1[index1] > nums2[index2]) {
                index2++;                                   //ͬ��
            } else {                                        //�ų�С�ڴ��������Ժ�
                intersection.push_back(nums1[index1]);      //���ظ�Ԫ��д�����
                index1++;
                index2++;
            }
        }
        return intersection;
    }
};
