from ultralytics import YOLO
import comet_ml
import os
import argparse

env = os.environ.copy()
# env["COMET_API_KEY"] = "MwCvlkPyBV0XqwuSoMFEM8GE8"
# os.environ.update(env)
comet_ml.init()
import argparse

# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description="参数示例")

# 添加命令行参数
parser.add_argument("--cfg", type=str, help="配置文件")
parser.add_argument("--weight", type=str, help="权重文件")
parser.add_argument("--data", type=str, help="数据文件")
parser.add_argument("--batch", type=int, help="批次大小")
parser.add_argument("--epochs", type=int, help="训练轮数")
parser.add_argument("--task", type=str, help="任务类型")
parser.add_argument("--device", type=int, help="设备号")
parser.add_argument("--freeze", type=int, help="冻结层数")
parser.add_argument("--optimizer", type=str, help="优化器类型")

# 设置默认值
parser.set_defaults(
    cfg="yolov8s.yaml",
    weight="weights/yolov8s.pt",
    data="ultralytics/cfg/datasets/VOC.yaml",
    batch=64,
    epochs=100,
    task="detect",
    device=0,
    freeze=10,
    optimizer="AdamW",
)

# 解析命令行参数
args = parser.parse_args()
model = YOLO(args.cfg).load(args.weight)
if __name__ == "__main__":
    # 训练模型
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        task=args.task,
        optimizer=args.optimizer,
        device=args.device,
        freeze=args.freeze,
    )
    # 模型验证
    model.val()
    print("Change successfully")
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "yolo_fb.pt")
    model.save(save_path)
