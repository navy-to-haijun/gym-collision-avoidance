# README

## 记录

### 2022/10/5

- 安装环境，跑通example

### 2022/10/7

* 逐条代码测试 example
* learning policy 指定要离散动作空间；
* 确定train过程中的可视化方法(保存每每个episode的图片)

## 2022/10/20

* trian 的时候使用RVO策略时无视learning policy

### 2022/10/21

* 使用ppo完成agent到达终点，无障碍物；

## 步骤

```shell
# clone 源码
git clone -b haijun --recursive git@github.com:navy-to-haijun/gym-collision-avoidance.git
# 创建环境
conda create -n robot python=3.7 
# 激活环境
conda activate robot
# 启动
conda activate robot
# 安装相关包
cd gym-collision-avoidance/
./install.sh
# 运行demo
cd gym_collision_avoidance/experiments/src/
python example
```

问题：

protobuf版本问题：

```bash
pip install protobuf==3.20
```

No module named 'scipy'

```bash
pip install scipy
```

降gym版本

```bash
pip install gym==0.15.7
```

No module named 'requests'

```bash
pip install requests
```



