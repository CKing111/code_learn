
/*
����һ���������� nums?���ҵ�һ���������͵����������飨���������ٰ���һ��Ԫ�أ������������͡�

ʾ�� 1��
���룺nums = [-2,1,-3,4,-1,2,1,-5,4]
�����6
���ͣ�����������?[4,-1,2,1] �ĺ����Ϊ?6 ��

ʾ�� 2��
���룺nums = [1]
�����1

ʾ�� 3��
���룺nums = [0]
�����0

ʾ�� 4��
���룺nums = [-1]
�����-1

ʾ�� 5��
���룺nums = [-100000]
�����-100000
*/

#include<iostream>
#include<vector>
using namespace std;


//��������
class Solution
{
public:
    int maxSubArray(vector<int> &nums)
    {
        //����Ѱ�������Сֵ����Ŀ����ʼֵһ��Ҫ����������ϵ���С���ֵ
        int max = INT_MIN;
        int numsSize = int(nums.size());        //��ȡ����
        for (int i = 0; i < numsSize; i++)          //����
        {
            int sum = 0;                            //��ʼ���ͽ��
            for (int j = i; j < numsSize; j++)      //�ӵ�ǰ�ʼ�������
            {   
                sum += nums[j];                     //��sum
                if (sum > max)                      //�Աȴ�С
                {
                    max = sum;                      //���������ڵ�ǰmax����ֵ
                }
            }
        }

        return max;
    }
};


//��̬�滮
class Solution {
public:
    int maxSubArray(vector<int>& nums) 
    {
        //��ʼ����Ĭ����Сֵ
        int r = nums.size();
        int Max = INT_MIN;
        //����f������f[i]��ʾ�ڵ�i����������2
        vector<int> f(r,0);
        //����ֵ
        f[0] = nums[0];
        Max = f[0];                                 //MaxΪ�������Ҫ������������ͽ�����ȸ���ʼֵ
        for(int i = 1;i < r;i++){                   //����
            f[i] = max(f[i-1]+nums[i],nums[i]);     //��̬�滮ת�ƺ�����ֻ������һ������
            Max = max(f[i],Max);                    //������
        }
        return Max;
    }
};

//��̬�滮--�Ż�
//����ת�ƺ���ֻ�õ���f[i]��f[i-1]����һ��һ������ת�ƣ����ֻ��Ҫ����һ��intֵ����ת�Ƽ�����������
class Solution {
public:
    int maxSubArray(vector<int>& nums) 
    {
        
        int r = nums.size();
        int Max = INT_MIN;
    
        // vector<int> f(r,0);
        // f[0] = nums[0];
        int f(nums[0]);
        Max = f;                                 
        for(int i = 1;i < r;i++){                   
            f = max(f+nums[i],nums[i]);     //ֻ�ı�������f  
            Max = max(f,Max);   
        }                 
        return Max;
    }
};

//̰���㷨
class Solution {
public:
    int maxSubArray(vector<int>& nums) 
    {
    /*
     * 	��ĿҪ������ֻ�����Ž⣬����ȡ����ֵ����ô�����׻ῼ�ǵ�̰���㷨
     * 	������Ҫ���ǵ�̰�������������tmpֵ�ӵ���ֵ(Ϊ0��ʵҲ������������Ϊû���ô�)��ʱ����ôǰ����Ӵ��ͺ�����ַ����ֻ����ɸ���Ӱ��(̰�ĸ�Ӱ��,ͨ�׵�˵����ǰ���Ӵ��ͺ�����ϻ�������汾���)��
     * 	��ˣ�����̰�ĵ�������ǰ����Ӵ������½����Ӵ������ֵ
     * */
        int n = nums.size();
    	int Max = nums[0];
    	//������ʱֵ
    	int tmp = 0;
    	for(int i = 0;i < n;i++) {                  //����.0-n
    		tmp += nums[i];                         //��ʱֵ������Ԫ�����
    		if(tmp > Max) Max = tmp;                //�жϵ�ǰֵ��������������ʹ�С��������ʱ���¸�ֵMax
    		if(tmp < 0) tmp = 0;                    //����͵���ʱֵС��0����������¿�ʼ
    	}                                           //��̰�ķ��У�����������ʱ����ǰ��Ϊ�������Ժ�������κΰ�����Ӧ����
    	return Max;
    }
};