/*
����һ���ַ������ҵ����ĵ�һ�����ظ����ַ���������������������������ڣ��򷵻� -1��

ʾ����

s = "leetcode"
���� 0

s = "loveleetcode"
���� 2

��ʾ������Լٶ����ַ���ֻ����Сд��ĸ��
*/ 

#include<iostream>
#include<vector>
#include<unordered_map>
using namespace std;

class Solution {
public:
    int firstUniqChar(string s) {
        unordered_map<int, int> frequency;
        //����s����д������map
        for(char map:s){
            ++frequency[map];
        }
        //����frequency������s�е�һ�������Ҵ���Ϊ1��Ԫ������
        for(int i = 0;i < s.size();i++){
            if (frequency[s[i] == 1]){
                return i;
            }
        }
        return -1;
    }
};

class Solution {
public:
    int firstUniqChar(string s) {
        unordered_map<int, int> position;
        int n = s.size();
        for (int i = 0; i < n; ++i) {
            if (position.count(s[i])) {
                position[s[i]] = -1;
            }
            else {
                position[s[i]] = i;
            }
        }
        int first = n;
        for (auto [_, pos]: position) {
            if (pos != -1 && pos < first) {
                first = pos;
            }
        }
        if (first == n) {
            first = -1;
        }
        return first;
    }
};

