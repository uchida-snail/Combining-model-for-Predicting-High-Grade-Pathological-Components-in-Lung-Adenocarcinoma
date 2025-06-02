# -*- coding: utf-8 -*-
r"""
针对患者文件夹下可能多层嵌套 DICOM 子目录的情况，递归查找所有包含 DICOM 序列的子文件夹，并将其转换为 .nii.gz。
"""

import os
import shutil
import pandas as pd
import SimpleITK as sitk


def read_patient_list_from_excel(excel_path: str, sheet_name=0) -> set:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    if "Patient" not in df.columns:
        raise ValueError("Excel 文件中未发现标题为 'Patient' 的列，请检查表头是否正确。")
    return set(df["Patient"].astype(str).str.strip().str.lower())


def find_patient_folders(dicom_root_dir: str, patient_set: set) -> dict:
    mapping = {}
    for entry in os.listdir(dicom_root_dir):
        full_path = os.path.join(dicom_root_dir, entry)
        if os.path.isdir(full_path):
            entry_name = entry.strip().lower()
            if entry_name in patient_set:
                mapping[entry_name] = full_path
    return mapping


def convert_dicom_folder_to_nii_recursive(patient_folder: str, output_folder: str, output_basename: str):
    """
    对 patient_folder 目录下的所有子目录递归查找 DICOM 序列（只要 GetGDCMSeriesIDs 返回非空，即视为包含 DICOM）。
    对每个找到的子目录，生成对应的 .nii.gz 文件，写入 output_folder。
    """
    temp_dir = r"C:\temp_nii"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    series_count = 0

    for root, dirs, files in os.walk(patient_folder):
        # 跳过空目录
        if not any(f.lower().endswith(".dcm") for f in files):
            # 如果文件名不以 .dcm 结尾，也可能是无后缀 DICOM，但先尝试 GetGDCMSeriesIDs
            pass

        # 尝试从本目录读取 series IDs
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
        if not series_IDs:
            continue

        # 如果本目录包含一个或多个序列，分别处理
        for idx, series_id in enumerate(series_IDs, start=1):
            file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, series_id)
            if not file_names:
                continue

            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(file_names)
            try:
                image3D = reader.Execute()
            except Exception as e:
                print(f"    ❌ `{root}` 序列 `{series_id}` 读取失败：{e}")
                continue

            # 构造唯一的文件名前缀：患者名 + 子目录相对路径（避免重名）
            relative_path = os.path.relpath(root, patient_folder).replace(os.sep, "_")
            if len(series_IDs) == 1 and relative_path == ".":
                base_name = output_basename
            else:
                series_count += 1
                rel = relative_path if relative_path != "." else f"Series{idx}"
                base_name = f"{output_basename}_{rel}"

            temp_output_path = os.path.join(temp_dir, base_name + ".nii.gz")
            try:
                sitk.WriteImage(image3D, temp_output_path)
            except Exception as e:
                print(f"    ❌ 写临时 `{temp_output_path}` 失败：{e}")
                continue

            final_output_path = os.path.join(output_folder, base_name + ".nii.gz")
            try:
                shutil.copyfile(temp_output_path, final_output_path)
                print(f"✔ `{root}` → 临时 `{temp_output_path}` → 最终 `{final_output_path}`")
            except Exception as e:
                print(f"    ❌ 复制 `{temp_output_path}` 到 `{final_output_path}` 失败：{e}")

    if series_count == 0:
        print(f"⚠ 警告：未在 `{patient_folder}` 及其子目录中找到任何 DICOM 序列。")


def main():
    excel_path      = r"C:\Users\21581\Desktop\250601内部验证集V2.xlsx"
    dicom_root_dir  = r"E:\241202内部测试集数据\WHC数据241201"
    output_root_dir = r"C:\Users\21581\Desktop\250601_internal_nii"

    try:
        patient_set = read_patient_list_from_excel(excel_path)
    except Exception as e:
        print(f"❌ 读取 Excel 文件失败：{e}")
        return

    print(f"读取 {len(patient_set)} 位患者：{patient_set}\n")

    mapping = find_patient_folders(dicom_root_dir, patient_set)
    if not mapping:
        print(f"⚠ 未在 `{dicom_root_dir}` 下找到匹配患者文件夹。")
        return

    print("匹配到以下患者文件夹：")
    for name_lower, folder_path in mapping.items():
        print(f"  - {name_lower}：{folder_path}")
    print()

    for name_lower, folder_path in mapping.items():
        output_basename = name_lower.replace(" ", "_")
        convert_dicom_folder_to_nii_recursive(
            patient_folder=folder_path,
            output_folder=output_root_dir,
            output_basename=output_basename
        )

    print("\n所有处理完成。")


if __name__ == "__main__":
    main()
