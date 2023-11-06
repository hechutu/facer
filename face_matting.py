import argparse
import os
import sys

import torch
from matplotlib import pyplot as plt
import facer
from tqdm import tqdm

sys.path.append('..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def single_process(type, filePath, outputPath):
    image = facer.hwc2bchw(facer.read_hwc(filePath)
                           ).to(device=device)  # image: 1 x 3 x h x w

    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        detectFaces = face_detector(image)

    face_parser = facer.face_parser('farl/celebm/448', device=device)  # optional "farl/lapa/448"
    with torch.inference_mode():
        faces = face_parser(image, detectFaces)

    seg_logits = faces['seg']['logits']

    '''
    plot algorithm img
    '''
    # seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    # n_classes = seg_probs.size(1)
    # vis_seg_probs = seg_probs.argmax(dim=1).float() / n_classes * 255
    # vis_img = vis_seg_probs.sum(0, keepdim=True)
    # facer.show_bhw(vis_img)
    # facer.show_bchw(facer.draw_bchw(image, faces))

    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

    vis_seg_probs = seg_probs.argmax(dim=1).float() * 255

    if 1 == type:
        max_value = findHairValue(vis_seg_probs)
        vis_seg_probs[vis_seg_probs == max_value] = 0
    vis_seg_probs[vis_seg_probs.cpu() != 0] = 255
    plt.imsave(outputPath, vis_seg_probs[0], cmap='gray')


def findHairValue(vis_seg_probs):
    temp = vis_seg_probs
    max_value = torch.max(temp)
    count = count_occurrences(temp, max_value)
    while count < 200:
        temp[temp == max_value] = 0
        max_value = torch.max(temp)
        count = count_occurrences(temp, max_value)
    return max_value

def count_occurrences(tensor, value):
    mask = torch.eq(tensor, value)
    count = torch.sum(mask).item()
    return count

def gen_images(config):
    inputDir = config.inputDir
    outputDir = config.outputDir
    type = config.type
    if not os.path.exists(inputDir):
        print("文件夹不存在")
        return

    if not os.path.exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)

    for filename in tqdm(os.listdir(inputDir)):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            input_path = os.path.join(inputDir, filename)
            ori_filename = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            new_filename = f"{ori_filename}-t{type}-mask{ext}"
            output_path = os.path.join(outputDir, new_filename)
            single_process(type, input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="face tool"
    )

    parser.add_argument(
        '--inputDir', '-i', type=str, help="输入目录"
    )
    parser.add_argument("--outputDir", '-o', type=str, help="输出目录")
    parser.add_argument("--type", '-t', type=int, help="类型,1-仅脸,2-脸和头发")

    args = parser.parse_args()
    print(f"inputDir path: {args.inputDir}")
    print(f"outputDir path: {args.outputDir}")
    print(f"type: {args.type}")

    # gen_images(args)
    single_process(1, 'input/10.300716.jpg', 'output/10.300716.jpg')
