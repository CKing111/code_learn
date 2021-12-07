/*
����һ���������� nums?��һ������Ŀ��ֵ target�������ڸ��������ҳ� ��ΪĿ��ֵ target? ����?����?���������������ǵ������±ꡣ
����Լ���ÿ������ֻ���Ӧһ���𰸡����ǣ�������ͬһ��Ԫ���ڴ��ﲻ���ظ����֡�
����԰�����˳�򷵻ش𰸡�

ʾ�� 1��
���룺nums = [2,7,11,15], target = 9
�����[0,1]
���ͣ���Ϊ nums[0] + nums[1] == 9 ������ [0, 1] ��

ʾ�� 2��
���룺nums = [3,2,4], target = 6
�����[1,2]

ʾ�� 3��
���룺nums = [3,3], target = 6
�����[0,1]
?

��ʾ��
2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
ֻ�����һ����Ч��
*/
#include<iostream>
#include<vector>
using namespace std;

//����ö��
//��������nums�����ÿһ��Ԫ�ص�����������������ݣ��ҵ�ǡ�ù���target������
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        for (int i = 0;i < nums.size()-1;i++){
            for(int j = i+1;j<nums.size();j++){
                if(nums[i] + nums[j] == target)
                    return {i,j};
            }
        }
        return {};
    }
};
/*
���Ǳ��������� aa ʱ���� targettarget ��ȥ aa���ͻ�õ� bb��
�� bb �����ڹ�ϣ���У����ǾͿ���ֱ�ӷ��ؽ���ˡ��� bb �����ڣ�
��ô������Ҫ�� aa �����ϣ�����ú�������������ʹ�á�
*/

#include<unordered_map>
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashtable;                  //�������ڹ�ϣ�������map������
                                                            //�����˰���<int,int>���͵ļ�ֵ��
        for (int i = 0; i < nums.size(); ++i) {
            auto it = hashtable.find(target - nums[i]);     //������ key Ϊ���ļ�ֵ�ԣ�����ҵ����򷵻�һ��ָ��ü�ֵ�Ե������������
            if (it != hashtable.end()) {
                return {it->second, i};     //it->first:�ü�ֵ�Ե�keyֵ
                                            //it->second:�ü�ֵ�Ե�value���  
            }
            hashtable[nums[i]] = i;         //��numsֵ�����д���ϣ��
        }
        return {};
    }
};

//�����ϣ��
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int,int> a;//����hash��������Ԫ��
        vector<int> b(2,-1);//��Ž��
        for(int i=0;i<nums.size();i++)                          //����nums�����뵽hash����
            a.insert(map<int,int>::value_type(nums[i],i));
        for(int i=0;i<nums.size();i++)
        {
            if(a.count(target-nums[i])>0&&(a[target-nums[i]]!=i))
            //�ж��Ƿ��ҵ�Ŀ��Ԫ����Ŀ��Ԫ�ز����Ǳ���
            {
                b[0]=i;
                b[1]=a[target-nums[i]];
                break;
            }
        }
        return b;
    };
};


//hash:�ַ���ת����
//�ַ�ֻ�д�д��26����
//�ַ��д�Сд����52����
#include<iostream>
#include<string>
//ֻ�д�д
int hashFunction1(char* S,int len){
    int id = 0;
    for(int i = 0;i < len;i++){
        id = id * 26 + (S[i] - 'A');
    }
    return id;
}
//��Сд
int hashFunction2(char* S ,int len){
    int id = 0;
    for(int i = 0;i < len;i++){
        if (S[i] >= 'A'&&S[i] <= 'Z')
        {
            id = id*52 + (S[i] - 'A');
        }
        else if (S[i] >= 'a'&&S[i] <= 'z')
        {
            id = id*52 + (S[i]-'a') + 26;
        }
    }
    return id;
}
//�����ڴ�Сд���ֵ��ַ���
//����ƴ�ӷ�������β����ƴ�ӵ��ַ�ת��������
int hashFunction3(char* S,int len){
    int id = 0;
    for(int i = 0;i < len;i++){
        id = id * 26 + (S[i] - 'A');
    }
    id = id*10 + (S[len -1] - '0');
    return id;
}
int main(){
    char str1[] = "ABC";
    char str2[] = "ABCa";
    char str3[] = "BCD4";
    std::cout<<hashFunction1(str1,3)<<std::endl;
    std::cout<<hashFunction2(str2,4)<<std::endl;
    std::cout<<hashFunction3(str3,4)<<std::endl;
    return 0;
}
//A:0 B:1 C:2 
//ABC =A*26^2 + B*26^1 + C*26^0