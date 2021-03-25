from PIL import Image, ImageDraw


def binary_img(my_img, threshold, binary_array):
    """
    二值化照片
    @param my_img: 未二值化的灰度图片
    @param threshold:  二值化阈值
    @param binary_array:  二值像素矩阵字典
    @return: binary_array
    """
    for x in range(0, my_img.size[0]):
        for y in range(0, my_img.size[1]):
            pix = my_img.getpixel((x, y))
            if pix >= threshold | (x == 0) | (y == 0):
                binary_array[(x, y)] = 1  # 白色
            else:
                binary_array[(x, y)] = 0  # 黑色


def denoise(my_img, binary_array, surround_pt_nums=3):
    """
    降噪
    @param my_img:  二值化的
    @param surround_pt_nums: 环绕的有效点数
    @param binary_array: 二值像素矩阵字典
    """
    for j in range(0, 5):
        for x in range(1, my_img.size[0]-1):
            for y in range(1, my_img.size[1]-1):
                surround_pt = 0
                if binary_array[(x, y)] == binary_array[(x-1, y)]:
                    surround_pt += 1
                if binary_array[(x, y)] == binary_array[(x-1, y-1)]:
                    surround_pt += 1
                if binary_array[(x, y)] == binary_array[(x-1, y+1)]:
                    surround_pt += 1
                if binary_array[(x, y)] == binary_array[(x+1, y)]:
                    surround_pt += 1
                if binary_array[(x, y)] == binary_array[(x+1, y-1)]:
                    surround_pt += 1
                if binary_array[(x, y)] == binary_array[(x+1, y+1)]:
                    surround_pt += 1
                if binary_array[(x, y)] == binary_array[(x, y-1)]:
                    surround_pt += 1
                if binary_array[(x, y)] == binary_array[(x, y+1)]:
                    surround_pt += 1

                if surround_pt < surround_pt_nums:
                    binary_array[(x, y)] = 1  # 噪点设置为白色


def save_img(filename, size, binary_array):
    """
    存储二值化图片
    @param filename: 图片文件名
    @param size: 图片尺寸
    @param binary_array: 二值像素矩阵字典
    @return:
    """
    my_img = Image.new("1", size)
    draw = ImageDraw.Draw(my_img)

    for x in range(size[0]):
        for y in range(size[1]):
            if (x == 0) or (y == 0) or (x == size[0]-1) or (y == size[1]-1):
                draw.point((x, y), 1)  # 去掉边缘上的噪声
            else:
                draw.point((x, y), binary_array[(x, y)])
    my_img.save(filename)


def handle_imgs():
    """
    处理所有图片，得到二值化后的图片(可以反复去噪)
    """
    for i in range(1, 500+1):
        img = Image.open('img_set/'+str(i)+'.png')
        img = img.convert('L')
        binary_array = {}

        binary_img(img, 180, binary_array)    # 二值化
        denoise(img, binary_array)         # 去噪
        save_img('img_binary_set/'+str(i)+'.png', img.size, binary_array)   # 保存去噪后的图片


def cut_imgs(cut_width=10, std_width=20):
    """
    切割图片
    @return:
    """
    threshold_nums = [0, 1, 2, 3, 4, 5, 6, 7]
    failed_cut = 0
    for i in range(1, 500+1):
        img = Image.open('img_fix_set/'+str(i)+'.png')  # 二值化后的图片
        cut_times = 0    # 切割次数
        cube_size = (std_width, img.size[1])     # 最后的训练集图片大小
        cursor = 0                                 # 切割游标
        for x in range(5, img.size[0]-1):            # 从开始靠后5的地方开始遍历
            crop_box = (cursor, 0, x, img.size[1])   # 切割块
            col_sum = get_col_sum(img, x)
            left_sum = get_col_sum(img, x-1)
            right_sum = get_col_sum(img, x+1)
            if (col_sum == 0) and (left_sum == 0):  # 与左边连续空白
                continue
            if x == img.size[0] - 2:                # 判定到越界了
                cut_times += 1
                std_img(img.crop(box=crop_box), cube_size, "img_crop_set/" + str(i) + '_' + str(cut_times) + '.png')
            # 类似于无穷符号的图像分布
            if (col_sum <= left_sum) and (col_sum <= right_sum) and (col_sum in threshold_nums):    # 如果最少黑点数在阈值内
                if (col_sum >= 5) and (not judge_disperse(img, x, col_sum)) and judge_lr(left_sum, col_sum, right_sum):
                    # 如果黑点过多，也不是离散分布, 或左右相差很小则判定不切割
                    continue
                if (x - cursor) < cut_width:  # 如果与上次切割点过近，则判定不切割, 即估计图片在宽度10以上
                    continue
                cut_times += 1
                std_img(img.crop(box=crop_box), cube_size, "img_crop_set/"+str(i)+'_'+str(cut_times)+'.png')
                cursor = x
            if cut_times >= 4:  # 如果进行了4次切割
                break
        if cut_times < 4:
            failed_cut += 1
    print(">>切割成功率：%.2f" % ((500 - failed_cut)/500*100)+"%")


def std_img(img, size, filename):
    my_img = Image.new("1", size)
    draw = ImageDraw.Draw(my_img)

    for x in range(size[0]):
        for y in range(size[1]):
            if (x >= img.size[0]) or (y >= img.size[1]):
                draw.point((x, y), 1)
            else:
                if img.getpixel((x, y)) > 128:
                    draw.point((x, y), 1)
                else:
                    draw.point((x, y), 0)
    my_img.save(filename)


def judge_lr(left, center, right):
    """
    判断邻近黑点数量
    @param left:
    @param center:
    @param right:
    @return:
    """
    if abs(left-center) <= 2 and abs(right-center) <= 2:
        return True
    else:
        return False


def judge_disperse(img, x, black_pts):
    """
    判断离散分布
    @param img:
    @param x:
    @param black_pts:
    @return:
    """
    disp_part = 0
    y = 0
    while y < img.size[1]:
        if img.getpixel((x, y)) < 128:
            disp_part += 1
            while y < img.size[1]:
                if img.getpixel((x, y)) < 128:
                    y += 1
                else:
                    break
        y += 1
    if (disp_part >= 2 and black_pts <= 6) or (disp_part >= 3 and black_pts >= 7):
        return True
    else:
        return False


def get_col_sum(img, x):
    """
    获取第x列的黑点数量
    @param img: 图片
    @param x: 列序号
    @return:
    """
    col_sum = 0
    for y in range(img.size[1]):
        if img.getpixel((x, y)) < 128:
            col_sum += 1
    return col_sum


def crop_blank():
    """
    去掉图片中多余的空白(因为每个字符的高度统一)
    @return:
    """
    for i in range(1, 500+1):
        img = Image.open('img_binary_set/'+str(i)+'.png')
        box = (5, 8, img.size[0]-1, img.size[1]-7)
        img.crop(box=box).save('img_shrink_set/'+str(i)+'.png')


def fix_img():
    """
    修复图片，让图片更健壮
    @return:
    """
    for i in range(1, 500+1):
        img = Image.open('img_shrink_set/'+str(i)+'.png')
        binary_array = {}
        # 再次二值化
        for x in range(0, img.size[0]):
            for y in range(0, img.size[1]):
                pix = img.getpixel((x, y))
                if pix >= 128:
                    binary_array[(x, y)] = 1  # 白色
                else:
                    binary_array[(x, y)] = 0  # 黑色
        # 修复
        for x in range(1, img.size[0]-1):
            for y in range(1, img.size[1]-1):
                # 对角线上的白点小于等于1个
                diag_blanks = binary_array[(x-1, y-1)] + binary_array[(x-1, y+1)] + binary_array[(x+1, y-1)] + binary_array[(x+1, y+1)]
                # 水平垂直上都是黑点
                un_diag_blanks = binary_array[(x, y-1)] + binary_array[(x, y+1)] + binary_array[(x-1, y)] + binary_array[(x+1, y)]
                if (diag_blanks <= 1) and (un_diag_blanks == 0) and (binary_array[(x, y)] == 1):  # 自己也是白点
                    binary_array[(x, y)] = 0
                    if binary_array[(x-1, y-1)] == 1:
                        binary_array[(x-1, y-1)] = 0
                    elif binary_array[(x-1, y+1)] == 1:
                        binary_array[(x-1, y+1)] = 0
                    elif binary_array[(x+1, y-1)] == 1:
                        binary_array[(x+1, y-1)] = 0
                    elif binary_array[(x+1, y+1)] == 1:
                        binary_array[(x+1, y+1)] = 0
                # 修复上下边缘
                if (y+1 == img.size[1]-1) or (y-1 == 0):
                    blacks = diag_blanks + un_diag_blanks
                    if (blacks == 1) and (binary_array[(x, y)] == 0):  # 只有一个白点且在边缘
                        binary_array[(x, y+1)] = 0
                        binary_array[(x, y-1)] = 0

        # 存储修复的图片
        my_img = Image.new("1", img.size)
        draw = ImageDraw.Draw(my_img)
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                draw.point((x, y), binary_array[(x, y)])
        my_img.save('img_fix_set/'+str(i)+'.png')


# handle_imgs()    # 获取二值化图片
# crop_blank()     # 剪掉多余的空白
# fix_img()        # 修复图片，健壮化
# cut_imgs()       # 图片分割