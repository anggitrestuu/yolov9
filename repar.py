import torch
from models.yolo import Model, DetectionModel

device = torch.device("cpu")
cfg = "./models/detect/gelan-c.yaml"
model = Model(cfg, ch=3, nc=4, anchors=3)
# model = model.half()
model = model.to(device)
_ = model.eval()

# Load checkpoint with weights_only=False to allow custom classes
ckpt = torch.load("./rdd2022/best.pt", map_location="cpu", weights_only=False)
model.names = ckpt["model"].names
model.nc = ckpt["model"].nc

idx = 0
for k, v in model.state_dict().items():
    if "model.{}.".format(idx) in k:
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 1))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt["model"].state_dict()[kr]
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace(
                "model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 16)
            )
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt["model"].state_dict()[kr]
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace(
                "model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 16)
            )
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt["model"].state_dict()[kr]
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace(
                "model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 16)
            )
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt["model"].state_dict()[kr]
    else:
        while True:
            idx += 1
            if "model.{}.".format(idx) in k:
                break
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 1))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt["model"].state_dict()[kr]
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace(
                "model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 16)
            )
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt["model"].state_dict()[kr]
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace(
                "model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 16)
            )
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt["model"].state_dict()[kr]
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace(
                "model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 16)
            )
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt["model"].state_dict()[kr]
_ = model.eval()

m_ckpt = {
    "model": model.half(),
    "optimizer": None,
    "best_fitness": None,
    "ema": None,
    "updates": None,
    "opt": None,
    "git": None,
    "date": None,
    "epoch": -1,
}
torch.save(m_ckpt, "./rdd2022/rdd-yolov9-c-converted.pt")


# python export.py --weights "./rdd/rdd-yolov9-c-converted.pt" --include onnx
