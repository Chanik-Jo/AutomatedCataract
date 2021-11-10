import glob

import cv2 as cv
import Augmentor,os,shutil

def resize_img(im):
    resized_img = cv.resize(im, (128, 128))

    return resized_img

    # https://github.com/mdbloice/Augmentor


def create_samples(dir):
    p = Augmentor.Pipeline(dir)
    # Point to a directory containing ground truth data.
    # Images with the same file names will be added as ground truth data
    # and augmented in parallel to the original data.
    # p.ground_truth("/path/to/ground_truth_images")
    # Add operations to the pipeline as normal:
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.flip_top_bottom(probability=0.5)
    p.sample(1000)


if __name__ == '__main__':
    dir_list = ['neg', 'pos']
    dest = "input_path"
    # lets create a sample of 500 negative and 500 positive Images
    for img_dir in dir_list:
        print("************************Processing {0} Directory*****************************".format(img_dir))
        create_samples(img_dir)
        # Now lets move files created to input_path folder
        source = img_dir + "/output/" #output는 augment적용이후 이동한 이미지들의 임시폴더.

        for img in glob.glob(source+"/*.jpg"):
            image=cv.imread(img)
            rl=resize_img(image)
            cv.imwrite(f'{img}',rl)

        files = os.listdir(source)
        for f in files:
            shutil.move(source + f, dest)#dest가 최종 destination.
        print("********************************************************************************")


#첫번째 소스의 기능은 랜덤으로 이미지들을 뻥튀기시켜 input_path에 넣는 코드이다.