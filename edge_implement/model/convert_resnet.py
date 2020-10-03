import torch
import torchvision
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--class_num', type=int, default=5,
                        help="Number of your last linear layer")
    parser.add_argument('--input_shape', type=str, nargs='*',
                        help="Input shape of your model")
    parser.add_argument('--pretrained', type=str,
                        help="Your pretrained model path")
    parser.add_argument('--target', type=str, help="Path for converted model")
    args = parser.parse_args()

    input_shape = [int(s) for s in args.input_shape[0].split(',')]
    # model = torchvision.models.resnet18()
    # model.fc = torch.nn.Linear(model.fc.in_features, args.class_num)
    # model.load_state_dict(torch.load(args.pretrained))

    model = torch.load(args.pretrained)
    print(model)
    model.maxpool = torch.nn.MaxPool2d(3, 2, dilation=1, ceil_mode=False)
    trace_data = torch.randn(input_shape)
    trace_model = torch.jit.trace(model.cpu().eval(), trace_data)
    torch.jit.save(trace_model, args.target)
    # torch.jit.save(trace_model, './ckpt/resent18.pt')


if __name__ == "__main__":
    main()
