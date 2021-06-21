
in_path = 'I:/Images_OCR/after_image/Voc_med/test.txt'
out_path = 'I:/Images_OCR/after_image/Voc_med/testvoc2coco.txt'

if __name__ == '__main__':
    f_open = open(in_path, 'r')  # 读取训练集
    f_write = open(out_path, 'w')  # 将替换后的写到新文本内
    lines = f_open.readlines()
    lists = ''
    for line in lines:
        rs = line.rstrip('\n')
        strs = rs.split(' ')
        lists += strs[1]+'\n'
    f_write.write(lists)
    f_open.close()
    f_write.close()
    print(in_path.split('/')[-1] + '=========》' + out_path.split('/')[-1] + '转化完毕')

