import pytorch_quantization as quant
import pytorch_quantization.nn as quant_nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# from https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/examples/calibrate_quant_resnet50.ipynb



def collect_quant_stats(model, data_loader: DataLoader, num_batches: int):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches, desc="Collecting quantization stats"):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_quant_amax(model, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, quant.calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
#             print(F"{name:40}: {module}")
    model.cuda()