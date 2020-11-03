# 데이터 디렉토리와 파일명 패턴을 입력받아
# tf 학습에 적절하도록 디렉토리 트리 형태로
# 변환시키는 코드

import argparse
import sys
import os
import shutil
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='images',
                    help="image data directory")
parser.add_argument('--pattern', default='_\d{4}_',
                    help="regex pattern of image files")


def main():
    # parse arguments
    args = parser.parse_args()
    img_root = args.dir
    pattern = re.compile(args.pattern)

    # check if already organized
    if False not in list(map(os.path.isdir, [img_root + '/' + f for f in os.listdir(img_root)])):
        print('=====================================================')
        print('ALREADY ORGANIZED. EXITING')
        print('=====================================================')

        sys.exit()

    # check if files are images
    if not glob.glob(img_root + '/' + '*.jpg') and not glob.glob(img_root + '/' + '*.png') and not glob.glob(img_root + '/' + '*.jpeg'):
        print('=====================================================')
        print('NO IMAGES FOUND. EXITING')
        print('=====================================================')

        sys.exit()

    print('=====================================================')
    print('STARTED ORGANIZING {} IMAGES'.format(len(os.listdir(img_root))))
    print('=====================================================')

    # reorganize images by regex pattern
    cnt = 0
    for file_name in os.listdir(img_root):
        dir_name = pattern.search(file_name).group()
        old_full_dir = os.path.join(img_root, file_name).replace('\\', '/')  # full old directory
        new_dir = os.path.join(img_root, dir_name).replace('\\', '/')
        new_full_dir = os.path.join(img_root, dir_name, file_name).replace('\\', '/')  # full new directory
        if not os.path.exists(new_dir):  # if directory does not exist, make it
            os.makedirs(new_dir)
            cnt += 1
        shutil.move(old_full_dir, new_full_dir)

    print('=====================================================')
    print('SUCCESSFULLY ORGANIZED INTO {} DIRECTORIES'.format(cnt) )
    print('=====================================================')

    # test if successful
    gen = ImageDataGenerator()
    test_gen = gen.flow_from_directory(img_root)

if __name__ == "__main__":
    main()