#include<iostream>

using namespace std;

// 1. ��̬����ת����
void test01() {

	// ������������
	char a = 'a';
	
	// static_cast<Ŀ������> (ԭ����)
	double d = static_cast<double>(a);

	cout << d << endl;
}

// �Զ�������
class Base {
	virtual void func() {};
};
class Son :public Base{
	virtual void func() {};		// ��̬��������д�����麯��
};
class Other {};
void test02() {
	// �Զ�����������
	Base* base = NULL;
	Son* son = NULL;

	// baseת��ΪSon����  --- ��������ת��������ȫ
	Son* son2 = static_cast<Son*>(base);
	// sonתΪBase* ����------��������ת������ȫ
	Base* base2 = static_cast<Son*>(son);
	cout << son2 << endl;

	// base ת��ΪOther*
	// ʧ�ܣ�û�и��ӹ�ϵ����������֮�����޷�ת���ɹ���
	//Other* oth = static_cast<Base*>(base);
}


// 2. ��̬����ת��
void test03() {
	// ������������
	// ʧ�ܣ���̬����ת��������������������֮���ת��
	//char c = 'c';
	//double d = dynamic_cast<double>(c);

	// �Զ�����������
	Base* base = NULL;
	Son* son = NULL;

	// base ת��ΪSon*���ͣ�����ȫ
	// ʧ�ܣ�����ȫת�����ɹ���ֻ�б�̶�̬�Ż�ת���
	//Son* son2 = dynamic_cast<Son*>(base);

	// son תΪ Base* , ��ȫ, �ɹ�
	Base* base2 = dynamic_cast<Base*>(son);

	// base ת��Ϊ Other*��ʧ�ܣ��Ǹ����޹�ϵ���޷���̬
	//Other* oth = dynamic_cast<Other*>(base);

	// ���������̬����ô����֮���ת�����ǰ�ȫ��
	Base* base3 = new Son;		// ��̬ʹ�÷�ʽ������ָ���������ָ���������
	// ��������ת������ȫ,basezת��Son*
	Son* son3 = dynamic_cast<Son*>(base3);

	/*
	��仰����˼�ǣ��������һ������ָ������ã�������ָ��һ���������Ҳ����ָ��һ���������������뽫��ת��Ϊ����ָ������ã������ȷ����ʵ������ָ��һ���������ģ��������õ�һ�����������ת�������磬�������һ�� Animal ���һ�� Dog �࣬Dog �� Animal �����࣬��ô���������д��

	Animal* a = new Dog(); // a ָ��һ�� Dog ���󣬿�����������ת��
	Dog* d = dynamic_cast<Dog*>(a); // d Ҳָ��ͬһ�� Dog ��������ת���ɹ�

	Animal* b = new Animal(); // b ָ��һ�� Animal ���󣬲�����������ת��
	Dog* e = dynamic_cast<Dog*>(b); // e Ϊ nullptr������ת��ʧ��
	*/
}

// 3. ����ת��
// const_cast<type-id>(expression)
// �����ԶԷ�ָ��ͷ�������const_cast
void test04() {
	// ָ��֮���ת��
	const int* p = NULL;
	// ��const int * ת��Ϊ int *
	int* p2 = const_cast<int*>(p);

	// ��p2 ת��Ϊ const int *����
	const int* p3 = const_cast<const int*>(p2);

	// ����֮���ת��
	const int a = 10;
	const int& aRef = a;	// ȡ a�ĵ�ַ

	// Const int & ת��Ϊ int &
	int& aRef2 = const_cast<int&>(aRef);
	
	// ��ָ�������ת��ʧ��
	//int b = const_cast<int>(a);
	
	/*
	����ת����������;��

	�����ڳ�����Ա�������޸ķǳ�����Ա����2��
	���Խ�ָ���������ָ�������ת��Ϊָ��ǳ��������ָ������ã��Ӷ������޸�ԭ�����ֵ1 4��
	����������ֻ����ԭ�������ǳ���ʱ�źϷ�������ᵼ��δ������Ϊ4��
	���Խ�ָ���ױ�����ָ�������ת��Ϊָ����ױ�����ָ������ã��Ӷ����Ժ���ԭ������ױ���3��
	*/
}

// 4. ���½���ת�������ȫ����������,reinterpret_cast<type-id>(expression)
// ���Խ�һ��ָ������õ�����ת��Ϊ������ȫ��ͬ�����ͣ�ͨ���ǲ����ݵ����͡�
void test05() {
	// int -> int*
	int a = 10;
	int* p = reinterpret_cast<int*>(a);

	// Base * -> Other * ;
	Base* base = NULL;
	Other* other = reinterpret_cast<Other*>(base);

	/*
	��������;��

	���Խ�һ��ָ��ת��Ϊһ���㹻����������ͣ����߽�һ����������ת��Ϊһ��ָ��3��
	���Խ�һ��ָ��ת��Ϊһ����֮�޹ص����ָ�룬���߽�һ����Աָ��ת��Ϊһ����֮�޹ص���ĳ�Աָ��1 2��
	������һЩ����ĳ����£����ϣ����1��λ����4��������Ӳ����صı���У�ʹ�����½���ת�����ı�ָ������õĵײ�����Ʊ�ʾ4��
	*/
}
int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}