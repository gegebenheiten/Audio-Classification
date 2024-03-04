from mobilenet import MobileNetV2
import onnx
import torch


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    torch_model = MobileNetV2(num_classes=12)

    model_path = '/home/lucas/ESC/cnn/model_only_on_esc_dataset/mobilenet_100.pth'

    torch_model.load_state_dict(torch.load(model_path))

    # set the model to inference mode
    torch_model.eval().to(device)

    # Input to the model
    x = torch.randn(1, 3, 128, 173).to(device)
    torch_out = torch_model(x)
    print(f"torch_out shape: {torch_out.shape}")

    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "audio_MobileNetV2_100.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )


def check():
    onnx_model = onnx.load('audio_MobileNetV2_100.onnx')

    onnx.checker.check_model(onnx_model)

    print('无报错，onnx模型载入成功')

    print(onnx.helper.printable_graph(onnx_model.graph))


if __name__ == '__main__':
    # main()
    check()
