# syfertext-test
安装syfertext并测试blog中的案例

## 步骤

- [安装syfertext](#安装syfertext)
- [启动测试代码](#启动测试代码)


## 安装syferttext
---
### 1.1激活pysyft环境
```
conda activate syft-test
```

### 1.2克隆syfertext
```
git clone git@github.com:OpenMined/SyferText.git
```
### 1.3cd到syfertext项目目录下

### 1.4切换分支并安装

```
pip install git+git://github.com/Nilanshrajput/syfertext_en_core_web_lg@master
git checkout update_additive_sharing
python setup.py install
```

## 启动测试代码
---

```
python main.py
```
