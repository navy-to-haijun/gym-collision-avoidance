# README

## 记录

### 2022/10/5

- 安装环境，跑通example

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



