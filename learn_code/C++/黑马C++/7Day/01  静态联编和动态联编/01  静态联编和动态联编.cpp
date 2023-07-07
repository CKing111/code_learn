#include<iostream>

using namespace std;

class Animal {
public:
	// ����virtual�ؼ��֣�speak���麯����
	virtual void speak() {
		cout << "������˵����" << endl;
	}

	virtual void eat(int a) {
		cout << "�����ڳԷ���" << endl;
	}
};

class Cat :public Animal {
public:
	// ��д�麯��
	void speak() {
		cout << "è��˵����" << endl;
	}
	void eat(int a) {
		cout << "Сè�ڳԷ���" << endl;
	}
};

class Dog :public Animal {
public:
	void speak() {
		cout << "����˵����" << endl;
	}
	void eat(int a) {
		cout << "С���ڳԷ���" << endl;
	}
};

// 1. �����м̳й�ϵ���࣬c++���Բ���ͨ������ǿת
// 2. ��̬����---��ַ�Ѿ��󶨣��Ѿ��޶������Ա����������������໹�ǻ���ø���
// 3. ��̬����---��ַû�а����������Ա�������麯��������virtual�ؼ���

// 4. ��̬������������
// 1��. �������麯����virtual��
// 2��. ���������д���麯������д��ָ����������ȫһ����
// 3��. �����ָ���������ָ������Ķ���eg��Animal & animal = Cat & cat
// 4��. ������д�����У����Բ���virtual


/*
�麯����̬��ָͨ������ָ�������ָ����������󣬲������麯��ʱ��ʵ�����е����������ʵ�֡�
���ֶ�̬�Կ��������ڻ����ж���ͨ�õĽӿں���Ϊ�������������Զ���ʵ��ϸ�ڣ��Ӷ�ʵ�ִ���������չ��ά����

ʵ���麯����̬��Ҫ�ڻ��ຯ��������ǰ���� virtual �ؼ��֣�
�����������ڽ��к�������ʱ�����ʵ�ʶ�������ѡ����ȷ�ĺ���ʵ�֣������Ǹ���ָ�����������ѡ����ʵ�֡�

class Cat       size(8):
		+---
 0      | +--- (base class Animal)
 0      | | {vfptr}
		| +---
		+---

Cat::$vftable@:
		| &Cat_meta
		|  0
 0      | &Cat::speak
*/
void doSpeak(Animal & animal) {
	animal.speak();
}
void doEat(Animal& animal) {
	animal.eat(10);
}

void test01() {
	Cat cat;
	Dog dog;

	doSpeak(cat);	// �ɹ���Animal & animal = Cat & cat��return�� Animal
	doSpeak(dog);	// �ɹ���Animal & animal = Cat & cat��return�� Animal

	doEat(cat);
	doEat(dog);
}

void test02() {
	Animal* animal = new Cat;		// ��̬������ָ������
	// �ײ�ʵ�֣��ȼ��ڣ�animal->speak();
	// *(int *)*(int *)animal   ��ʾCat::speak�����ڱ��еĵ�ַ
	((void(*)())(*(int*)*(int*)animal))();
	/*
	������
		1����һ����*(int *)*(int *)animal��ͨ��ָ�����㾫ȷ��λCat��speak����������������еĵ�ַ
			����C++�Ķ���ģ�ͣ��������麯�����࣬���Ķ����д洢һ��ָ���������ָ�룬Ҳ��Ϊ��ָ�롣
			����ǽ������������麯���ĵ�ַ���ն���˳������һ�����У���ָ��ָ��ñ����ʼ��ַ��
			�������������¶��常����麯��ʱ����������в������µ�ʵ�ֵ�ַ�����Ǹ����е�ԭ��ַ���Ӷ�ʵ�ֶ�̬�ԡ�
			��ˣ�ͨ��������animalָ�������ת������ת��Ϊһ��int����ָ�룬���ɵõ�Cat�������ָ��ָ�������ַ��
			Ȼ���ٴν����øõ�ַ���õ�Cat������������speak��������ڵ�ַ
		2���ڶ�������Cat��speak��������ڵ�ַת��Ϊ�޲��޷���ֵ�ĺ���ָ�룺(void(*)())...
			���ڸ�ָ����Ҫ�������ĺ������ã����Ҫ��Cat��speak��������ڵ�ַת��Ϊһ��ָ�����ͣ�
			��ָ�����ͱ�ʾһ���޲��޷���ֵ�ĺ����������ͨ��C++�е�ָ������ǿ��ת���﷨ʵ�֡�
			��󣬽�Cat��speak��������ڵ�ַת��Ϊ�޲��޷���ֵ�ĺ���ָ���ʹ�ú��������﷨����ִ�У�
			����ʵ�ֶ�Cat��speak��Ա�����ĵ��á�
	*/

	// ʵ��animal->eat();
	/*
	class Cat       size(8):
        +---
 0      | +--- (base class Animal)
 0      | | {vfptr}
        | +---
        +---

Cat::$vftable@:
        | &Cat_meta
        |  0
 0      | &Cat::speak
 1      | &Cat::eat

	*/
	//((void(*)())* ((int*)*(int*)animal + 1))();

	// C++Ĭ�ϵ��ù�����cdecl���ָ���__stdcall
	typedef void(__stdcall* FUNC)(int);
	(FUNC(*((int*)*(int*)animal + 1)))(10);
}

int main() {
	test01();

	test02();

	cout << sizeof(Animal) << endl;
	system("pause");
	return EXIT_SUCCESS;


}