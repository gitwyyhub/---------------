import torch
from model import ResNet
from option import get_args

opt = get_args()


def pth_to_onnx():
    args = opt

    device = torch.device('cpu')

    model = ResNet(num_classes=3, pretrained=False)
    ckpt = torch.load(args.test_model_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, args.loadsize, args.loadsize)

    input_names = ['input']
    output_names = ['output']

    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print('ONNX model saved to: {}'.format(args.onnx_path))


if __name__ == '__main__':
    pth_to_onnx()