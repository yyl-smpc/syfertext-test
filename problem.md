# 真实环境问题

## pysyft

### 安装版本及方式
```shell
pip install syft==0.2.8 \\用此命令安装syft 0.2.8版本  
```

## pygrid
### 安装版本及方式

#### 1.下载官方源码
```shell
git clone git@github.com:OpenMined/PyGrid.git
```

#### 2.切换分支并制作镜像
```shell
git checkout master
docker build ./apps/node/ -f ./apps/node/Dockerfile -t pygrid:node \\制作node节点image
docker build ./apps/network/ -f ./apps/network/Dockerfile -t pygrid:network \\制作network节点image
```

#### 3.修改compose.yaml文件

openmined/grid-network:production改为pygrid:network

openmined/grid-node:production改为pygrid:node

note:node节点并没有注册到network，所有测试代码用的privatenetwork方式
#### 4.启动集群
```shell
docker-compose up
```

## syfertext
## 安装版本及方式
#### 1.下载官方源码
```shell
git clone git@github.com:OpenMined/SyferText.git
```

#### 2.切换分支并安装
```shell
pip install git+git://github.com/Nilanshrajput/syfertext_en_core_web_lg@master
git checkout update_additive_sharing
python setup.py install
```

## 测试代码及问题
测试用的代码即本源码的real分支中的main.py

### 1.出现的错误
![img](./error.png)