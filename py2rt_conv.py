import torchvision
import torchvision.models as models
import torch
from torch2trt import torch2trt
print("-----Pytorch to TensorRT conversion for inference-----")
model = models.resnet18(pretrained=False, num_classes=8).eval().cuda()
model.load_state_dict(torch.load('model_resnet18_new_data_2.pth'))
print("Model loaded")
x = torch.ones((1, 3, 480, 640)).cuda()
model_trt = torch2trt(model, [x], fp16_mode=True)
print(model_trt, "\nTesting TRT Module:")
y = model(x)

y_trt = model_trt(x)
print(torch.max(torch.abs(y - y_trt)))
print('Saving model')
torch.save(model_trt.state_dict(), 'resnet18_torch2rt_1602.pth')
