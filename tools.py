import os


def convert_bbox(bbox, img_size):
    image_width = img_size[0]
    image_height = img_size[1]

    # 遍历所有标注

    x_center, y_center, width, height = bbox

    # 计算左上角和右下角的坐标
    x_top = x_center * image_width - (width * image_width) / 2
    y_top = y_center * image_height - (height * image_height) / 2
    x_bottom = x_top + width * image_width
    y_bottom = y_top + height * image_height
    return [x_top, y_top, x_bottom, y_bottom]
    # print(f"Class ID: {class_id}, Top Left: ({x_top}, {y_top}), Bottom Right: ({x_bottom}, {y_bottom})")


def get_labels(label_path, img_size):
    # 还没有计算bbox，暂时以原格式返回
    res = []
    # name = os.path.join(path, filename)
    with open(label_path, 'r') as file:
        for line in file:
            res.append(line.split())
            res[-1][0]=int(res[-1][0])
            res[-1][1:] = [float(item) for item in res[-1][1:]]
            res[-1][1:] = convert_bbox(res[-1][1:], img_size)
    return res


def get_gt(label_path, img_size):
    tmp = get_labels(label_path, img_size)
    gt = []
    for cnt, it in enumerate(tmp):
        gt.append({'image_id': cnt,
                   'category_id': it[0],
                   'bbox': it[1:]})

    return gt

# filename = './datasets/test.txt'
# tmp = get_gt(filename)
