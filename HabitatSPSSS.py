# advanced_habitat_cluster_analysis_fixed.py

"""
完整示例脚本（Second Fix）：
- 针对九个簇（labels 1–9）提取多种图像特征：密度、纹理、形态、分形、位置、簇质量
- 簇间统计分析（Kruskal-Wallis + Bonferroni）
- 可视化：箱线图、Violin 图、Heatmap、PCA 散点图
- 自动生成 report.pdf，将所有 PNG 图整合

修复内容：
- 移除对 3D 情况下 rp.eccentricity 的调用，直接设为 NaN，避免 NotImplementedError
- 其余功能不变
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk

from scipy.stats import skew, kurtosis, iqr, kruskal
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import regionprops, label as sklabel
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# -------------------------------------------
# 用户需根据自己环境修改以下路径：
# -------------------------------------------
ROI_FOLDER    = r"E:\peritumor\folder\output"  # ROI 文件目录
IMAGE_PARENT  = r"E:\peritumor\ITH"            # CT 图像根目录（每个患者子文件夹，以 "<patientID>_" 命名）
OUTPUT_DIR    = r"E:\peritumor\ClusterReport"  # 输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------
# 辅助函数
# -------------------------------------------
def load_nifti(path):
    """
    通过 nibabel 加载 NIfTI，返回 numpy 数组及 affine。
    """
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine

def find_roi(pid):
    """
    在 ROI_FOLDER 中查找名称为 '<pid>__habitat.nii.gz' 的文件。
    """
    pattern = os.path.join(ROI_FOLDER, f"{pid}__habitat.nii.gz")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def compute_fractal_dimension(bin3d):
    """
    使用盒计数法（Minkowski–Bouligand）近似计算三维分形维数。
    """
    max_dim = min(bin3d.shape) // 2
    sizes = 2 ** np.arange(1, int(np.floor(np.log2(max_dim))) + 1)
    counts = []
    for s in sizes:
        nx, ny, nz = bin3d.shape
        count = 0
        for x in range(0, nx, s):
            for y in range(0, ny, s):
                for z in range(0, nz, s):
                    block = bin3d[x:x+s, y:y+s, z:z+s]
                    if block.any():
                        count += 1
        counts.append(count)
    if len(sizes) < 2:
        return np.nan
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# -------------------------------------------
# 特征提取函数：针对单个簇 mask，提取所需特征
# -------------------------------------------
def extract_cluster_features(ct_arr, mask_arr, affine, tumor_mask):
    """
    提取单簇特征，包括：
      1) 第一阶强度：mean, median, std, skew, kurt, IQR, P10, P90, CoV, hist_energy, hist_entropy
      2) GLCM（二阶纹理）：contrast, correlation, energy, homogeneity
      3) GLSZM（高阶纹理）：LZE, ZP（可选，若可调用则计算，否则设 NaN）
      4) 3D 形态：volume_mm3, surface_area_mm2, sphericity, surf_vol_ratio, eccentricity (3D 不支持, 设 NaN),
         compactness, fractal_dim, radial_dist_var
      5) 位置先验：簇质心 vs 肿瘤中心 欧氏距离
      6) 簇质量（Silhouette, Davies-Bouldin），此处设 NaN，可根据需求扩展
    """
    # ----------
    # a) 第一阶强度
    # ----------
    voxels = ct_arr[mask_arr > 0]
    if voxels.size == 0:
        return None

    feats = {}
    feats['mean_hu']   = float(np.mean(voxels))
    feats['median_hu'] = float(np.median(voxels))
    feats['std_hu']    = float(np.std(voxels))
    feats['skew_hu']   = float(skew(voxels))
    feats['kurt_hu']   = float(kurtosis(voxels))
    feats['iqr_hu']    = float(iqr(voxels))
    feats['p10_hu'], feats['p90_hu'] = np.percentile(voxels, [10, 90])
    feats['cov_hu']    = float(feats['std_hu'] / (feats['mean_hu'] + 1e-8))

    hist, _ = np.histogram(voxels, bins=32, density=True)
    hist = hist + 1e-8
    feats['hist_energy']  = float(np.sum(hist ** 2))
    feats['hist_entropy'] = float(-np.sum(hist * np.log(hist)))

    # ----------
    # b) GLCM 特征（2D 中间切片）
    # ----------
    coords = np.column_stack(np.where(mask_arr > 0))
    mid_z = int(np.median(coords[:, 2]))
    slice2d = ct_arr[:, :, mid_z]
    mask2d  = mask_arr[:, :, mid_z]

    if mask2d.any():
        sl_min, sl_max = slice2d.min(), slice2d.max()
        if sl_max > sl_min:
            scaled = ((slice2d - sl_min) / (sl_max - sl_min) * 255).astype(np.uint8)
        else:
            scaled = np.zeros_like(slice2d, dtype=np.uint8)
        img2d = np.where(mask2d, scaled, 0).astype(np.uint8)
        glcm = greycomatrix(img2d,
                            distances=[1],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256,
                            symmetric=True,
                            normed=True)
        feats['glcm_contrast'] = float(np.mean(greycoprops(glcm, 'contrast')))
        feats['glcm_corr']     = float(np.mean(greycoprops(glcm, 'correlation')))
        feats['glcm_energy']   = float(np.mean(greycoprops(glcm, 'energy')))
        feats['glcm_homog']    = float(np.mean(greycoprops(glcm, 'homogeneity')))
    else:
        feats['glcm_contrast'] = np.nan
        feats['glcm_corr']     = np.nan
        feats['glcm_energy']   = np.nan
        feats['glcm_homog']    = np.nan

    # ----------
    # c) GLSZM 特征（可选）
    #    动态导入 gray_level_size_zone_matrix，若不可用则跳过
    # ----------
    try:
        from skimage.feature import gray_level_size_zone_matrix
        glszm_2d = gray_level_size_zone_matrix(img2d, levels=256, mask=mask2d)
        zones = np.nonzero(glszm_2d)
        if glszm_2d.sum() > 0:
            i_inds = np.arange(glszm_2d.shape[0])[:, None, None]
            lze = float(np.sum((i_inds ** 2) * glszm_2d) / (glszm_2d.sum() + 1e-8))
            zp  = float(len(zones[0]) / (glszm_2d.sum() + 1e-8))
        else:
            lze = np.nan
            zp  = np.nan
        feats['glszm_LZE'] = lze
        feats['glszm_ZP']  = zp
    except ImportError:
        feats['glszm_LZE'] = np.nan
        feats['glszm_ZP']  = np.nan

    # ----------
    # d) 3D 形态特征
    # ----------
    bin3d = (mask_arr > 0).astype(np.uint8)
    labeled = sklabel(bin3d)
    rp_list = regionprops(labeled)
    if rp_list:
        rp = rp_list[0]
        spacing = np.abs(affine[:3, :3].diagonal())
        voxel_vol = float(np.prod(spacing))

        vox_count = rp.area
        feats['volume_mm3'] = float(vox_count * voxel_vol)

        # 表面积：SimpleITK LabelContourImageFilter
        sitk_mask = sitk.GetImageFromArray(bin3d)
        sitk_mask.SetSpacing(spacing.tolist())
        contour = sitk.LabelContourImageFilter().Execute(sitk_mask)
        contour_arr = sitk.GetArrayFromImage(contour)
        feats['surface_area_mm2'] = float(np.count_nonzero(contour_arr) * spacing[0] * spacing[1])

        vol = feats['volume_mm3']
        sa  = feats['surface_area_mm2']
        feats['sphericity']     = float((np.pi ** (1/3) * (6 * vol) ** (2/3)) / (sa + 1e-8))
        feats['surf_vol_ratio'] = float(sa / (vol + 1e-8))
        # 3D 情况下，不支持 rp.eccentricity，直接设为 NaN
        feats['eccentricity']   = np.nan  
        feats['compactness']    = float((sa ** 3) / (36 * np.pi * (vol ** 2))) if vol > 0 else np.nan
        feats['fractal_dim']    = float(compute_fractal_dimension(bin3d))

        coords3d = coords.astype(float)
        tumor_coords = np.column_stack(np.where(tumor_mask))
        tumor_center = tumor_coords.mean(axis=0)
        dists = np.linalg.norm(coords3d - tumor_center, axis=1)
        feats['radial_dist_var'] = float(np.var(dists))
    else:
        feats.update({
            'volume_mm3': np.nan, 'surface_area_mm2': np.nan,
            'sphericity': np.nan, 'surf_vol_ratio': np.nan,
            'eccentricity': np.nan, 'compactness': np.nan,
            'fractal_dim': np.nan, 'radial_dist_var': np.nan
        })

    # ----------
    # e) 位置先验：簇质心 vs 肿瘤中心 欧氏距离
    # ----------
    cluster_center = coords.mean(axis=0)
    tumor_coords = np.column_stack(np.where(tumor_mask))
    tumor_center = tumor_coords.mean(axis=0)
    feats['center_dist'] = float(np.linalg.norm(cluster_center - tumor_center))

    # ----------
    # f) 簇质量（Silhouette, Davies-Bouldin），设为 NaN
    # ----------
    feats['silhouette'] = np.nan
    feats['db_index']   = np.nan

    return feats

# -------------------------------------------
# 主流程：遍历所有患者，提取 Feature
# -------------------------------------------
all_rows = []
errors = []

for patient_folder in tqdm(os.listdir(IMAGE_PARENT), desc="Processing Patients"):
    if not patient_folder.endswith("_"):
        continue
    pid = patient_folder.rstrip("_")
    ct_path = os.path.join(IMAGE_PARENT, patient_folder, "image_path.nii")
    roi_path = find_roi(pid)
    if not (os.path.isfile(ct_path) and roi_path):
        errors.append((pid, "Missing CT or ROI file"))
        continue

    # 加载 CT + ROI
    ct_arr, affine = load_nifti(ct_path)
    roi_arr, _     = load_nifti(roi_path)
    tumor_mask     = roi_arr > 0

    # 对 labels 1–9 依次提取特征
    for lbl in range(1, 10):
        mask_arr = (roi_arr == lbl).astype(np.uint8)
        feats = extract_cluster_features(ct_arr, mask_arr, affine, tumor_mask)
        if feats is None:
            continue
        feats['patient_id']     = pid
        feats['cluster_label']  = lbl
        all_rows.append(feats)

# 合并为 DataFrame 并保存到 CSV
df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(OUTPUT_DIR, "all_clusters_features.csv"), index=False)

# -------------------------------------------
# 簇间统计分析：Kruskal-Wallis + Bonferroni 校正
# -------------------------------------------
stats_list = []
feature_names = [c for c in df.columns if c not in ['patient_id', 'cluster_label']]
for fname in feature_names:
    groups = [sub[fname].dropna().values for _, sub in df.groupby("cluster_label")]
    if any(len(g) == 0 for g in groups):
        continue
    H, p = kruskal(*groups)
    stats_list.append({'feature': fname, 'H_stat': float(H), 'p_value': float(p)})
stats_df = pd.DataFrame(stats_list)
stats_df['p_adjusted'] = stats_df['p_value'] * len(stats_df)  # Bonferroni 校正
stats_df.to_csv(os.path.join(OUTPUT_DIR, "stats_summary.csv"), index=False)

# -------------------------------------------
# 可视化：箱线图、violin 图、Heatmap、PCA
# -------------------------------------------
plot_dir = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(plot_dir, exist_ok=True)

# 1. Violin 图示例：mean_hu
plt.figure(figsize=(10, 6))
sns.violinplot(x="cluster_label", y="mean_hu", data=df, inner="box", palette="viridis")
plt.title("Distribution of Mean HU by Cluster")
plt.xlabel("Cluster Label")
plt.ylabel("Mean HU")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "violin_mean_hu.png"), dpi=300)
plt.close()

# 2. 箱线图示例：std_hu
plt.figure(figsize=(10, 6))
sns.boxplot(x="cluster_label", y="std_hu", data=df, palette="viridis")
plt.title("Boxplot of HU Std Dev by Cluster")
plt.xlabel("Cluster Label")
plt.ylabel("HU Std Dev")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "box_std_hu.png"), dpi=300)
plt.close()

# 3. Heatmap: 各簇各特征中位数
median_df = df.groupby("cluster_label")[feature_names].median().T
plt.figure(figsize=(12, 10))
sns.heatmap(median_df, cmap="viridis", annot=False)
plt.title("Median Feature Values per Cluster")
plt.xlabel("Cluster Label")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "heatmap_median_features.png"), dpi=300)
plt.close()

# 4. PCA 散点图：选取部分关键特征
selected_feats = ['mean_hu', 'std_hu', 'glcm_contrast', 'glcm_energy',
                  'volume_mm3', 'sphericity']
selected_feats = [f for f in selected_feats if f in df.columns]
if len(selected_feats) >= 2:
    X = df[selected_feats].fillna(0).values
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=df["cluster_label"],
                    palette="tab10", s=20)
    plt.title("PCA of Selected Cluster Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pca_selected_features.png"), dpi=300)
    plt.close()

# -------------------------------------------
# 生成 PDF 报告：将所有 PNG 图像合并
# -------------------------------------------
pdf_path = os.path.join(OUTPUT_DIR, "cluster_report.pdf")
c = canvas.Canvas(pdf_path, pagesize=A4)
page_w, page_h = A4

png_files = sorted(glob.glob(os.path.join(plot_dir, "*.png")))
y_pos = page_h - 50

for img_path in png_files:
    if y_pos < 300:
        c.showPage()
        y_pos = page_h - 50
    try:
        img = plt.imread(img_path)
        aspect = img.shape[0] / img.shape[1]
        display_w = page_w - 60
        display_h = display_w * aspect
        c.drawImage(img_path, 30, y_pos - display_h, width=display_w, height=display_h)
        y_pos -= (display_h + 20)
    except Exception:
        continue

c.save()

# -------------------------------------------
# 最终输出信息
# -------------------------------------------
print(f"Features extracted for {len(df['patient_id'].unique())} patients, total clusters: {df.shape[0]}")
print(f"Feature CSV: {os.path.join(OUTPUT_DIR, 'all_clusters_features.csv')}")
print(f"Stats summary: {os.path.join(OUTPUT_DIR, 'stats_summary.csv')}")
print(f"Figures in: {plot_dir}")
print(f"PDF report: {pdf_path}")
if errors:
    err_df = pd.DataFrame(errors, columns=["patient_id", "reason"])
    err_df.to_csv(os.path.join(OUTPUT_DIR, "error_log.csv"), index=False)
    print(f"Some patients skipped; see error_log.csv for details.")
