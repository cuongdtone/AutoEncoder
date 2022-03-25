import cv2
import numpy as np
def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)
    return img
def concat_image(list_image, grid_shape = (2, 8), image_size=(112, 112*2)):
    rows, collums = grid_shape
    out_image = np.zeros((rows*image_size[1], collums*image_size[0], 3)).astype('uint8')
    for idx, image in enumerate(list_image):
        x_position = (idx % collums) * image_size[0]
        y_position = (idx // collums) * image_size[1]
        out_image[y_position:y_position+image_size[1], x_position:x_position+image_size[0], :] = image
    out_image = draw_grid(out_image, grid_shape, color=(255, 255, 255), thickness=1)
    return out_image

if __name__ == '__main__':
    image = concat_image(1)
    cv2.imshow('out', image)
    cv2.waitKey()
