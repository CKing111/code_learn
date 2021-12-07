/*
给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

示例：

s = "leetcode"
返回 0

s = "loveleetcode"
返回 2

提示：你可以假定该字符串只包含小写字母。
*/ 

#include<iostream>
#include<vector>
#include<unordered_map>
using namespace std;

class Solution {
public:
    int firstUniqChar(string s) {
        unordered_map<int, int> frequency;
        //历遍s将其写入无序map
        for(char map:s){
            ++frequency[map];
        }
        //历遍frequency，读出s中第一个出现且次数为1的元素索引
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

