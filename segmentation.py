import cv2
# import matplotlib.pyplot as plt
import os
import shutil
import numpy as np

SEG_IMG_NAME = 'segmented.png'
DIR_NAME = 'segmented_chars'

def rows_text_present(img, rows, columns):
    row_intensities = []
    for row in rows:
        zero_pixels = False
        for column in columns:
            if img[row][column] == 255:
                zero_pixels = True
                break
        row_intensities.append(zero_pixels)
    return row_intensities

def make_dir(dir_name):
    if(os.path.exists(DIR_NAME)):
        shutil.rmtree(DIR_NAME)
    os.mkdir(DIR_NAME)

def draw_border(img, **kwargs):
    o = kwargs["left_col"]
    q = kwargs["right_col"]
    vert_up = kwargs["top_row"]
    vert_down = kwargs["bottom_row"]
    color = kwargs.get("color", (0, 255, 0))
    thickness = kwargs.get("thickness", 1)
    cv2.line(img, (o, vert_up), (o, vert_down), color, thickness)
    cv2.line(img, (q, vert_up), (q, vert_down), color, thickness)
    cv2.line(img, (o, vert_up), (q, vert_up), color, thickness)
    cv2.line(img, (o, vert_down), (q, vert_down), color, thickness)

def make_square(im, ds):
    desired_size = ds
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im


import time
def segment(input_img, **kwargs):
    segmented_imgs = []
    save_letters = kwargs.get("save_letters", False)
    # grayscale_img = kwargs.get("grayscale_img", True)
    orig_seg_out = kwargs.get("orig_seg_out", False)
    seg_img_size = kwargs.get("seg_img_size", 50)
    seg_img_num = 1
    start_time = time.time()
    if(isinstance(input_img,str)):
        grayscale = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
    else:
        grayscale = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(grayscale,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    end_time = time.time()
    rows, columns = thresh.shape
    rows = range(rows)
    columns = range(columns)
    row_intensities = rows_text_present(thresh, rows, columns)
    img_for_extraction = thresh
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    i = 0
    j = -1
    if save_letters:
        make_dir(DIR_NAME)
    for i in rows[1:-1]:
        if row_intensities[i]:
            if j == -1:
                j = i
        else:
            if j != -1:
                k = i
                o = -1
                vert_up = k
                vert_down = j
                for m in columns[1:-1]:
                    text_present = False
                    for n in range(j, k+1):
                        if all(x == 255 for x in thresh[n][m]):
                            text_present = True
                            vert_up = min(vert_up, n)
                            for d in reversed(range(j, k+1)):
                                if all(x == 255 for x in thresh[d][m]):
                                    vert_down = max(vert_down, d)
                                    break
                            break
                    if text_present:
                        if o == -1:
                            o = m
                    else:
                        if o != -1:
                            q = m
                            seg_img = img_for_extraction[vert_up-1:vert_down+1, o-1:q+1]
                            seg_img = make_square(seg_img, seg_img_size)
                            # _, seg_img = cv2.threshold(seg_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            segmented_imgs.append(seg_img)
                            if save_letters:
                                seg_img_path = "{0}/char_{1}.png".format(DIR_NAME, seg_img_num)
                                cv2.imwrite(seg_img_path, seg_img)
                            seg_img_num += 1
                            if orig_seg_out:
                                draw_border(thresh, left_col=o, right_col=q, top_row=vert_up,
                                            bottom_row=vert_down)
                            o = -1
                            vert_up = k
                            vert_down = j
                j = -1
    if orig_seg_out:
        cv2.imwrite(SEG_IMG_NAME, thresh)
    imgs = np.array(segmented_imgs)
    return imgs


if __name__ == "__main__":
    j = 0
    directory = "checkbox"
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            array = segment(directory + "\\" + filename)
            if not os.path.exists("test_set\\"+directory):
                os.makedirs("test_set\\"+directory)
            if (len(array) > 0):
                for img in array:
                    cv2.imwrite("test_set\\"+directory + "\\" + directory+ str(j)+ ".png", img)
                    j+=1
        else:
            continue

