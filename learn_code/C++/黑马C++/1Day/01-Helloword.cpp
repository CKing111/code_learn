#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};
    
    // cout 标准输出
    // << 左移运算符
    for (const string& word : msg){
        cout << word << " "<<endl;            
    }
    printf("END!!!"); 
    cout << endl;
    system("pause");    //阻塞功能
    return EXIT_SUCCESS;    //返回正常输出
}


