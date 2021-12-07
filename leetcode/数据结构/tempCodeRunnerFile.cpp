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