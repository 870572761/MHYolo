from ultralytics import YOLO
import itertools
import torch
import copy
import os


def find_different_parameters(obj1, obj2):
    different_parameters = []
    value1 = {}
    value2 = {}
    # 获取对象的参数列表
    parameters1 = vars(obj1)
    parameters2 = vars(obj2)
    # 遍历参数列表，比较参数值
    for parameter in parameters1.keys():
        # if 'Dict' in str(type(parameters1[parameter])):
        #     print(str(type(parameters1[parameter])), parameter)
        if "Dict" in str(type(parameters1[parameter])):
            sp, sv1, sv2 = find_different_parameters(parameters1[parameter], parameters2[parameter])
            for k, v1, v2 in zip(sp, sv1, sv2):
                different_parameters.append(parameter + k)
                value1[parameter + k] = v1
                value2[parameter + k] = v2
        elif isinstance(parameters1[parameter], torch.Tensor):
            if not torch.equal(parameters1[parameter], parameters2[parameter]):
                different_parameters.append(parameter)
                value1[parameter] = parameters1[parameter]
                value2[parameter] = parameters2[parameter]
        elif not parameters1[parameter].__eq__(parameters2[parameter]):
            different_parameters.append(parameter)
            value1[parameter] = parameters1[parameter]
            value2[parameter] = parameters2[parameter]
    return different_parameters, value1, value2


def check_p(m1, m2, s11, s12, s21, s22):
    f = True
    for i, j in zip(range(s11, s12), range(s21, s22)):
        for mo1, mo2 in zip(m1.model.model[i].children(), m2.model.model[j].children()):
            p, v1, v2 = find_different_parameters(mo1, mo2)
            for k in p:
                if k in "track_running_stats":
                    print("Just BN stats diff")
                else:
                    print(k, "mo1", i, v1[k], "mo1", j, v2[k])
                    f = False
                    break
    return f


import argparse

# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description="参数示例")

# 添加命令行参数
parser.add_argument("--rescfg", type=str, help="配置文件")
parser.add_argument("--weight1", type=str, help="权重文件原")
parser.add_argument("--weight2", type=str, help="权重文件新")
parser.add_argument("--save-dir", type=str, help="保存地址")

# 设置默认值
parser.set_defaults(
    rescfg="yolov8slei.yaml",
    weight1="weights/yolov8s.pt",
    weight2="runs/detect/train18/weights/best.pt",
    save_dir="weights",
)

# 解析命令行参数
args = parser.parse_args()

# Load a model
model1 = YOLO(args.weight1)  # pretrained YOLOv8n model
model2 = YOLO(args.weight2)  # 新加入的权重
modelres = copy.deepcopy(model2)  # 目标模型
modelt = YOLO(args.rescfg)  # 目标模型，备份用于后续恢复一些重要参数

# 超参数承接
modelres.model.model = copy.deepcopy(modelt.model.model)
modelres.model.save = copy.deepcopy(modelt.model.save)
print(model1.task, model2.task, modelres.task, modelt.task)
s1 = len(model1.model.model)  # 原层数
s2 = len(modelres.model.model)  # 现层数
idx2 = s1 * 2 - s2 + 1  # backbone后的第一层
print(s1, s2, idx2)
for i in range(10):
    modelres.model.model[i] = copy.deepcopy(model1.model.model[i])
    modelres.model.model[i].f = modelt.model.model[i].f
for i in range(10, s1):
    modelres.model.model[i] = copy.deepcopy(model1.model.model[i])
    modelres.model.model[i].f = modelt.model.model[i].f
for i in range(s1, s2 - 1):
    modelres.model.model[i] = copy.deepcopy(model2.model.model[idx2])
    modelres.model.model[i].f = modelt.model.model[i].f
    idx2 = idx2 + 1

# 检查m1与m2的backbone是否相等
f = [True, True, True, True, True]
f[0] = check_p(model1, model2, 0, 10, 0, 10)
# 检查mr与m1的backbone是否相等
f[1] = check_p(modelres, model1, 0, 10, 0, 10)
# 检查mr与m2的backbone是否相等
f[2] = check_p(modelres, model2, 0, 10, 0, 10)
# 检查mr与m1的head是否相等
f[3] = check_p(modelres, model1, 10, 23, 10, 23)
# 检查mr与m2的head是否相等
print(len(modelres.model.model), len(model2.model.model))
f[4] = check_p(modelres, model2, 23, 36, 10, 23)
fr = True
# 超参数承接
# 名字改一下
modelres.model.names = copy.deepcopy(model1.model.names)
modelres.model.yaml["nc"] = model1.model.yaml["nc"] + model2.model.yaml["nc"]
for i, det in model2.model.names.items():
    i = i + len(model2.model.names)
    modelres.model.names[i] = det
print(modelres.model.names)
for i in range(5):
    fr = fr & f[i]
if not fr:
    print("Something wrong")
else:
    print("Change successfully")
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "yolomhead.pt")
    modelres.save(save_path)
