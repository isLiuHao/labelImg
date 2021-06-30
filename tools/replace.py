import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_path',default='I:/Images_OCR/after_image/Voc_med/test.txt')
    parser.add_argument('--out_path',default='I:/Images_OCR/after_image/Voc_med/testvoc2coco.txt')
    args = parser.parse_args()
    f_open = open(args.in_path, 'r')  # 读取训练集
    f_write = open(args.out_path, 'w')  # 将替换后的写到新文本内
    lines = f_open.readlines()
    lists = ''
    for line in lines:
        rs = line.rstrip('\n')
        strs = rs.split(' ')
        lists += strs[1]+'\n'
    f_write.write(lists)
    f_open.close()
    f_write.close()
    print(args.in_path.split('/')[-1] + '=========》' + args.out_path.split('/')[-1] + '转化完毕')

