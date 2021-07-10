import os

# 找到images和annotation中多余的
if __name__ == "__main__":
    annotations_path = r'I:/Images_OCR/after_image/Voc_med/annotations'
    images_path = r'I:/Images_OCR/after_image/Voc_med/images'
    different_lists = []
    annotations_files = os.listdir(annotations_path)
    images_files = os.listdir(images_path)
    lena = 0
    lenb = 0
    while True:
        if lena == len(annotations_files)-1:
            break
        annotations_file = annotations_files[lena].split('.')[0]
        images_file = images_files[lenb].split('.')[0]
        if annotations_file != images_file:
            lena += 1
            different_lists.append(annotations_file)
        else:
            lena += 1
            lenb += 1
    print(different_lists)
