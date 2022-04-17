/*
给定一个整数数组 nums?和一个整数目标值 target，请你在该数组中找出 和为目标值 target? 的那?两个?整数，并返回它们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
你可以按任意顺序返回答案。

示例 1：
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

示例 2：
输入：nums = [3,2,4], target = 6
输出：[1,2]

示例 3：
输入：nums = [3,3], target = 6
输出：[0,1]
?

提示：
2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
只会存在一个有效答案
*/
#include<iostream>
#include<vector>
using namespace std;

//暴力枚举
//历遍两次nums，求出每一个元素的所有两向量求和数据，找到恰好构成target的索引
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
我们遍历到数字 aa 时，用 targettarget 减去 aa，就会得到 bb，
若 bb 存在于哈希表中，我们就可以直接返回结果了。若 bb 不存在，
那么我们需要将 aa 存入哈希表，好让后续遍历的数字使用。
*/

#include<unordered_map>
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashtable;                  //声明基于哈希表的无序map容器、
                                                            //声明了包含<int,int>类型的键值对
        for (int i = 0; i < nums.size(); ++i) {
            auto it = hashtable.find(target - nums[i]);     //查找以 key 为键的键值对，如果找到，则返回一个指向该键值对的正向迭代器；
            if (it != hashtable.end()) {
                return {it->second, i};     //it->first:该键值对的key值
                                            //it->second:该键值对的value序号  
            }
            hashtable[nums[i]] = i;         //将nums值和序号写入哈希表
        }
        return {};
    }
};

//两遍哈希表
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int,int> a;//建立hash表存放数组元素
        vector<int> b(2,-1);//存放结果
        for(int i=0;i<nums.size();i++)                          //历遍nums，插入到hash表中
            a.insert(map<int,int>::value_type(nums[i],i));
        for(int i=0;i<nums.size();i++)
        {
            if(a.count(target-nums[i])>0&&(a[target-nums[i]]!=i))
            //判断是否找到目标元素且目标元素不能是本身
            {
                b[0]=i;
                b[1]=a[target-nums[i]];
                break;
            }
        }
        return b;
    };
};


//hash:字符串转整数
//字符只有大写用26进制
//字符有大小写，用52进制
#include<iostream>
#include<string>
//只有大写
int hashFunction1(char* S,int len){
    int id = 0;
    for(int i = 0;i < len;i++){
        id = id * 26 + (S[i] - 'A');
    }
    return id;
}
//大小写
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
//若存在大小写数字的字符串
//采用拼接方法，将尾数字拼接到字符转换整数后
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