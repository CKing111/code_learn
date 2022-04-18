#include "ros/ros.h"
        //ros/ros.h是一个很便利的include，它包括了使用ROS系统中最常见的公共部分所需的全部头文件。
#include "std_msgs/String.h"
        //从std_msgs包里的String.msg文件中自动生成的头文件。

#include <sstream>

/**
 * This tutorial demonstrates simple sending of messages over the ROS system.
 */
int main(int argc, char **argv)
/*
                     * argc 是 argument count的缩写，表示传入main函数的参数个数,我们输入的main函数的参数个数应该是argc-1个；
                    * argv 是 argument vector的缩写，表示传入main函数的参数序列或指针，并且第一个参数argv[0]一定是程序的名称，并且包含了程序所在的完整路径，
*/
{
  /**
   * The ros::init() function needs to see argc and argv so that it can perform
   * any ROS arguments and name remapping that were provided at the command line.
   * For programmatic remappings you can use a different version of init() which takes
   * remappings directly, but for most command-line programs, passing argc and argv is
   * the easiest way to do it.  The third argument to init() is the name of the node.
   *
   * You must call one of the versions of ros::init() before using any other
   * part of the ROS system.
   */
  ros::init(argc, argv, "talker");                  
                            /**
                             * Init()的第三个参数是节点的名称。 
                             初始化ROS。这使得ROS可以通过命令行进行名称重映射——不过目前不重要。这也是我们给节点指定名称的地方。
                             节点名在运行的系统中必须是唯一的。注意：名称必须是基本名称，例如不能包含任何斜杠'/'。
                             **/

  /**
   * NodeHandle is the main access point to communications with the ROS system.
   * The first NodeHandle constructed will fully initialize this node, and the last
   * NodeHandle destructed will close down the node.
   */
  ros::NodeHandle n;
                    /*
                    为这个进程的节点创建句柄。创建的第一个NodeHandle实际上将执行节点的初始化，而最后一个被销毁的NodeHandle将清除节点所使用的任何资源。
                    */
  /**
   * The advertise() function is how you tell ROS that you want to
   * publish on a given topic name. This invokes a call to the ROS
   * master node, which keeps a registry of who is publishing and who
   * is subscribing. After this advertise() call is made, the master
   * node will notify anyone who is trying to subscribe to this topic name,
   * and they will in turn negotiate a peer-to-peer connection with this
   * node.  advertise() returns a Publisher object which allows you to
   * publish messages on that topic through a call to publish().  Once
   * all copies of the returned Publisher object are destroyed, the topic
   * will be automatically unadvertised.
   *
   * The second parameter to advertise() is the size of the message queue
   * used for publishing messages.  If messages are published more quickly
   * than we can send them, the number here specifies how many messages to
   * buffer up before throwing some away.
   */
  ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
                        /*
                        告诉主节点我们将要在chatter话题上发布一个类型为std_msgs/String的消息。
                        这会让主节点告诉任何正在监听chatter的节点，我们将在这一话题上发布数据。
                        第二个参数是发布队列的大小。在本例中，如果我们发布得太快，它将最多缓存1000条消息，不然就会丢弃旧消息。

                        NodeHandle::advertise()返回一个ros::Publisher对象，它有2个目的：
                        其一，它包含一个publish()方法，可以将消息发布到创建它的话题上；
                        其二，当超出范围时，它将自动取消这一宣告操作。
                        */
  ros::Rate loop_rate(10);
          //ros::Rate对象能让你指定循环的频率。
          //它会记录从上次调用Rate::sleep()到现在已经有多长时间，并休眠正确的时间。在本例中，我们告诉它希望以10Hz运行。
  /**
   * A count of how many messages we have sent. This is used to create  a unique string for each message.
   */
  int count = 0;
  while (ros::ok())
            /*
            ros::ok()在以下情况会返回false：

                          收到SIGINT信号（Ctrl+C）
                           被另一个同名的节点踢出了网络
                          ros::shutdown()被程序的另一部分调用
                           所有的ros::NodeHandles都已被销毁
            */
  {
    /**
     * This is a message object. You stuff it with data, and then publish it.
     */
    std_msgs::String msg;       //声明一个消息自适应的类

    std::stringstream ss;       //声明字符串
    ss << "hello world " << count;      //写入字符
    msg.data = ss.str();           //写入msg广播文件

    ROS_INFO("%s", msg.data.c_str());
                //ROS_INFO和它的朋友们可用来取代printf/cout。
        /**
         * 向 Topic: chatter 发送消息, 发送频率为10Hz（1秒发10次）；消息池最大容量1000。
         */
    chatter_pub.publish(msg);

    ros::spinOnce();
            //ROS消息回调处理函数:若有订阅者，此函数接受回调信息  
            //ros::spin() 或 ros::spinOnce()，两者区别在于前者调用后不会再返回，
            //也就是你的主程序到这儿就不往下执行了，而后者在调用后还可以继续执行之后的程序。
    loop_rate.sleep();      //达到10Hz的发布速率。
    ++count;
  }


  return 0;
}