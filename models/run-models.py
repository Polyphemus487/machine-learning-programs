import os

models = ['VGG19', 'MobileNetV3_Large', 'SqueezeNet1.0', 'AlexNet', 'darknet53', 'InceptionV3', 'Xception']


def run_through_models():
    for model in models:
        print(model)
        os.system(f'python imagenet_multi_generalized.py --model {model} --input-fldr /Users/sam/imagenet' \
                  +f'-vid-robust-example-master/imagenet-vid-robust/val --save-to-file /Users/sam/save-results/{model}'\
                  +'full --display-in-terminal False')


run_through_models()
