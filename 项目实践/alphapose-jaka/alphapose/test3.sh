#! /bin/bash
source ~/.bashrc
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/cxking/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/cxking/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/cxking/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/cxking/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# #启动虚拟环境
conda activate alphapose

#设置arm姿态索引的全局变量供python脚本使用
arm=1


# # gnome-terminal  --window -e 'bash -c "python test_python_json1.py;exec bash"' --window -e 'bash -c "roslaunch jaka_ros_driver start.launch;exec bash"' --window -e 'bash -c "sleep 20s;rostopic list;exrc bash"' --window -e 'bash -c "sleep 20s;rostopic list;exrc bash"'　

#启动ros-jaka

{
    gnome-terminal -t "ros" -x bash -c "roslaunch jaka_ros_driver start.launch;exec bash"
}&
sleep 10s
#循环
int=1
while(( $int<=5 ))
do
        
    #启动关节检测
    sleep 2s
    {
        gnome-terminal -t "alphapose" -x bash -c "python test_python_json3.py;exec bash"
    }&&
    echo $arm
    sleep 12s
    {
        gnome-terminal -t "roscall-move" -x bash -c "./test_arm_move2.sh;exec bash"
        sleep 1s
        # exit
    }


    let "int++"
done








#     # sleep 20s
#     if [ $ARM -gt '0' ];     #大于0
#     then
#         echo "Arm up !"
#         # ./test_arm_up.sh
#         sleep 10s
#         {
#             gnome-terminal -t "roscall-up" -x bash -c "./test_arm_up.sh;exec bash"
#         }
#     elif [ $ARM -eq '0' ];   #等于0
#     then
#         echo "Arm level！"
#         # ./test_arm_lev.sh
#         sleep 10s
#         {
#             gnome-terminal -t "roscall-lev" -x bash -c "./test_arm_lev.sh;exec bash"
#         }
#     elif [ $ARM -lt '0' ];
#     then
#         echo "Arm hem ！"
#         # ./test_arm_hem.sh
#         sleep 10s
#         {
#             gnome-terminal -t "roscall-hem" -x bash -c "./test_arm_hem.sh;exec bash"
#         }
#     else
#     echo "没有符合的条件"
#     fi
#     sleep 30s
#     let "int++"
# done




# rosservice call /robot_driver/move_joint "pose: [${x},${y},-1.654319252462968,2.261601485528601,1.60328049944293,0.8329127855351778]
# has_ref: false
# ref_joint: [0]
# mvvelo: 2.0
# mvacc: 0.5
# mvtime: 0.0
# mvradii: 0.0
# coord_mode: 0
# index: 0"


# sleep 20s
# {
# 	gnome-terminal -t "roscall" -x bash -c "/home/cxking/jaka_ros_ws/test_sh/test_joint2.sh;exec bash"
# }&

