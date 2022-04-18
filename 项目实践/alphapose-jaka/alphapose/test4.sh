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
#启动ros-jaka
{
    gnome-terminal -t "ros" -x bash -c "roslaunch jaka_ros_driver start.launch;exec bash"
}&
sleep 8s

#启动alphapose的python脚本
# sleep 2s
{
    gnome-terminal -t "alphapose" -x bash -c "python test_python_json4.py;exec bash"
}&&

#循环执行
int=1
while(( $int<=10 ))
do  
    #启动关节检测
    sleep 5s
    {
        gnome-terminal -t "roscall-move" -x bash -c "./test_arm_move2.sh;exec bash"
        sleep 5s

    }&&
    let "int++"
done

#exit