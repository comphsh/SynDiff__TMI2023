import os
import numpy
import random
import json
from copy import deepcopy

from timm.models.nfnet import test_nfnet

random.seed(42)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_train_test(patient_list, idx_list, train_all_num, test_num):
    json_path = r"/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/merge_MICCAI_BraTS2020_TrainingData"
    with open(os.path.join(json_path, "dataset_info_all.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)


    training = []
    testing = []
    data_dict = deepcopy(data)
    for idx in idx_list[0:train_all_num]:
        training.append(
            {
                'image': f"image/{patient_list[idx]}.nii",
                'label': f"label/{patient_list[idx]}.nii",
            }
        )

    for idx in idx_list[train_all_num:]:
        testing.append(
            {
                'image': f"image/{patient_list[idx]}.nii",
                'label': f"label/{patient_list[idx]}.nii",
            }
        )

    data_dict["numTraining"] = train_all_num
    data_dict["numTesting"] = test_num

    data_dict["training"] = training
    data_dict["testing"] = testing

    with open(os.path.join(json_path, "dataset_info_train8_test2.json"), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

def save_train_val_test(patient_list, idx_list, train_num, val_num, test_num):
    json_path = r"/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/merge_MICCAI_BraTS2020_TrainingData"
    with open(os.path.join(json_path, "dataset_info_all.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)


    training = []
    validation = []
    testing = []

    train_all_num = train_num + val_num

    data_dict = deepcopy(data)
    for idx in idx_list[0:train_num]:
        training.append(
            {
                'image': f"image/{patient_list[idx]}.nii",
                'label': f"label/{patient_list[idx]}.nii",
            }
        )

    for idx in idx_list[train_num:train_all_num]:
        validation.append(
            {
                'image': f"image/{patient_list[idx]}.nii",
                'label': f"label/{patient_list[idx]}.nii",
            }
        )

    for idx in idx_list[train_all_num:]:
        testing.append(
            {
                'image': f"image/{patient_list[idx]}.nii",
                'label': f"label/{patient_list[idx]}.nii",
            }
        )

    data_dict["numTraining"] = train_num
    data_dict["numValidation"] = val_num
    data_dict["numTesting"] = test_num

    data_dict["training"] = training
    data_dict["validation"] = validation
    data_dict["testing"] = testing

    with open(os.path.join(json_path, "dataset_info_train7_val1_test2.json"), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

def divide_to_train_test_val():
    test_ratio = 0.2
    train_ratio = 0.7
    val_ratio = 0.1

    path = "/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    patient_list = [item for item in os.listdir(path) if 'csv' not in item]
    print(len(patient_list))


    test_num = int(len(patient_list) * test_ratio)
    train_all_num = len(patient_list) - test_num

    val_num = int(len(patient_list) * val_ratio)
    train_num = train_all_num - val_num


    idx_list = list(range(0 , len(patient_list)))
    # [297, 248, 178, 267, 34, 231, 358, 209, 322, 289, 177, 91, 224, 53, 237, 360, 65, 106, 143, 74, 190, 271, 249, 320, 217, 72, 10, 233, 100, 103, 167, 147, 85, 280, 120, 259, 298, 343, 162, 170, 80, 329, 26, 363, 208, 240, 55, 227, 89, 77, 66, 25, 154, 230, 99, 38, 197, 206, 213, 158, 182, 244, 338, 299, 341, 96, 266, 225, 94, 146, 122, 37, 109, 19, 88, 303, 127, 313, 149, 139, 368, 201, 133, 211, 118, 283, 215, 191, 221, 330, 188, 212, 11, 247, 69, 325, 123, 246, 262, 200, 9, 321, 268, 311, 310, 179, 349, 276, 357, 113, 144, 31, 348, 198, 169, 43, 156, 171, 7, 366, 203, 291, 159, 6, 93, 130, 59, 105, 290, 90, 345, 165, 350, 48, 104, 272, 62, 180, 278, 331, 173, 340, 269, 60, 131, 151, 84, 8, 264, 175, 354, 18, 1, 257, 157, 58, 243, 315, 5, 86, 250, 367, 319, 30, 115, 132, 263, 207, 347, 102, 333, 305, 265, 54, 261, 210, 254, 242, 42, 365, 121, 168, 326, 32, 296, 17, 124, 187, 21, 20, 145, 318, 14, 61, 256, 92, 228, 4, 292, 82, 153, 223, 245, 199, 288, 41, 218, 95, 39, 50, 155, 336, 163, 76, 160, 27, 129, 45, 195, 128, 67, 222, 238, 0, 324, 307, 352, 75, 284, 294, 164, 196, 68, 192, 344, 226, 239, 29, 184, 253, 2, 241, 220, 141, 64, 270, 306, 251, 97, 312, 316, 152, 275, 335, 293, 337, 78, 56, 24, 46, 252, 260, 70, 286, 314, 204, 219, 134, 126, 362, 323, 73, 234, 202, 255, 277, 281, 108, 33, 137, 205, 161, 355, 117, 28, 166, 342, 285, 138, 304, 236, 300, 364, 273, 87, 36, 136, 107, 181, 189, 83, 186, 232, 334, 328, 51, 351, 317, 148, 116, 23, 35, 98, 295, 185, 150, 282, 40, 193, 63, 274, 235, 22, 135, 309, 176, 183, 49, 194, 353, 361, 172, 110, 79, 339, 174, 356, 81, 3, 142, 301, 229, 112, 214, 359, 332, 101, 287, 13, 308, 258, 119, 111, 47, 15, 16, 216, 302, 44, 279, 346, 52, 71, 114, 125, 140, 12, 57, 327]

    random.shuffle(idx_list)
    print(idx_list)
    print(len(idx_list))

    print(SCRIPT_DIR)
    # for comparison methods
    with open(os.path.join(SCRIPT_DIR, "train_all.list"), "w") as f:
        for idx in idx_list[0:train_all_num]:
            f.write(patient_list[idx]+"\n")

    with open(os.path.join(SCRIPT_DIR, "train.list"), "w") as f:
        for idx in idx_list[0:train_num]:
            f.write(patient_list[idx]+"\n")

    with open(os.path.join(SCRIPT_DIR, "val.list"), "w") as f:
        for idx in idx_list[train_num:train_all_num]:
            f.write(patient_list[idx]+"\n")

    with open(os.path.join(SCRIPT_DIR, "test.list"), "w") as f:
        for idx in idx_list[train_all_num:]:
            f.write(patient_list[idx]+"\n")

    # for my diff + moe
    # 保存为 trainall test
    save_train_test(patient_list, idx_list, train_all_num=train_all_num, test_num=test_num)

    # 保存为 train val test
    save_train_val_test(patient_list, idx_list, train_num=train_num, val_num=val_num, test_num=test_num)

if __name__ == "__main__":
    divide_to_train_test_val()
