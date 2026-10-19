"""QA figures for preprocess.py: a mid-slice dashboard and a multi-slice strip.

Diffusion volumes are resampled onto the T2 grid for display only, so overlays show
the physical alignment between the structural and diffusion outputs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def save_compact_qa(t1c, t1, t2, b0, dwi, adc, struct_mask, diff_mask, patient_id, output_path):
    resampler_z = sitk.ResampleImageFilter()
    resampler_z.SetReferenceImage(t2)
    resampler_z.SetInterpolator(sitk.sitkLinear)
    resampler_z.SetDefaultPixelValue(0.0)

    b0_mapped = resampler_z.Execute(b0)
    dwi_mapped = resampler_z.Execute(dwi)
    adc_mapped = resampler_z.Execute(adc)

    # Map the diffusion mask so it properly blanks FOV mismatches
    resampler_mask = sitk.ResampleImageFilter()
    resampler_mask.SetReferenceImage(t2)
    resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_mask.SetDefaultPixelValue(0)
    diff_mask_mapped = resampler_mask.Execute(diff_mask)

    arr_t1c = sitk.GetArrayViewFromImage(t1c)
    arr_t1  = sitk.GetArrayViewFromImage(t1) if t1 is not None else None
    arr_t2  = sitk.GetArrayViewFromImage(t2)
    arr_b0  = sitk.GetArrayViewFromImage(b0_mapped)
    arr_dwi = sitk.GetArrayViewFromImage(dwi_mapped)
    arr_adc = sitk.GetArrayViewFromImage(adc_mapped)
    arr_s_mask = sitk.GetArrayViewFromImage(struct_mask)
    arr_d_mask = sitk.GetArrayViewFromImage(diff_mask_mapped)

    z_valid = np.where(np.any(arr_b0 > 1e-5, axis=(1, 2)))[0]
    z_mid = z_valid[len(z_valid) // 2] if len(z_valid) > 0 else arr_t2.shape[0] // 2

    def get_slc(arr): return np.flipud(np.fliplr(arr[z_mid])) if arr is not None else None

    slc_t1c, slc_t1, slc_t2 = map(get_slc, [arr_t1c, arr_t1, arr_t2])
    slc_b0, slc_dwi, slc_adc = map(get_slc, [arr_b0, arr_dwi, arr_adc])
    slc_s_mask = get_slc(arr_s_mask)
    slc_d_mask = get_slc(arr_d_mask)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.patch.set_facecolor('black')
    fig.suptitle(f"QA Dashboard: Patient {patient_id} (Mid-FOV Slice Z:{z_mid})", color="white", fontsize=20, fontweight='bold')
    axes = axes.flatten()

    def format_ax(ax, title, image, is_overlay=False, vmin=None, vmax=None, mask_slc=None):
        if image is None:
            ax.axis('off'); ax.set_facecolor('black')
            ax.text(0.5, 0.5, "Not Available", color="gray", ha="center", va="center", transform=ax.transAxes, fontsize=16)
            ax.set_title(title, color="gray", fontsize=14)
            return

        if not is_overlay and mask_slc is not None and vmin is not None:
            display_img = np.where(mask_slc > 0.5, image, vmin)
        else:
            display_img = image

        if is_overlay:
            ax.imshow(display_img)
        else:
            ax.imshow(display_img, cmap="gray", vmin=vmin, vmax=vmax)
            if mask_slc is not None:
                ax.contour(mask_slc, levels=[0.5], colors=['#FFFF00'], linewidths=1.2, alpha=0.7)

        txt_fx = [PathEffects.withStroke(linewidth=2.5, foreground='black')]
        ax.text(image.shape[1]/2, 15, 'A', color='#00FFFF', fontsize=14, fontweight='bold', ha='center', va='top', path_effects=txt_fx)
        ax.text(image.shape[1]/2, image.shape[0]-15, 'P', color='#00FFFF', fontsize=14, fontweight='bold', ha='center', va='bottom', path_effects=txt_fx)
        ax.text(15, image.shape[0]/2, 'R', color='#00FFFF', fontsize=14, fontweight='bold', ha='left', va='center', path_effects=txt_fx)
        ax.text(image.shape[1]-15, image.shape[0]/2, 'L', color='#00FFFF', fontsize=14, fontweight='bold', ha='right', va='center', path_effects=txt_fx)
        ax.set_title(title, color="white", fontsize=16, pad=10, path_effects=txt_fx)
        ax.axis('off')

    def get_w(slc, mask_slc, is_diff=False):
        if slc is None or mask_slc is None: return -2, 2
        valid = slc[mask_slc > 0.5]
        if len(valid) == 0: return -2, 2
        p_low = np.percentile(valid, 2 if is_diff else 1)
        p_high = np.percentile(valid, 95 if is_diff else 99)
        return (-2, 2) if p_high - p_low < 1e-8 else (p_low, p_high)

    def norm_rgb(slc, mask_slc, is_diff=False):
        if slc is None or mask_slc is None: return None
        valid = slc[mask_slc > 0.5]
        if len(valid) == 0: return np.zeros_like(slc)
        p1 = np.percentile(valid, 1)
        p99 = np.percentile(valid, 95 if is_diff else 99)
        if p99 - p1 < 1e-8: return np.zeros_like(slc)
        normed = np.clip((slc - p1) / (p99 - p1), 0, 1)
        normed[mask_slc < 0.5] = 0.0
        return normed

    w_t1c = get_w(slc_t1c, slc_s_mask)
    w_t1  = get_w(slc_t1, slc_s_mask)
    w_t2  = get_w(slc_t2, slc_s_mask)

    w_b0  = get_w(slc_b0, slc_d_mask, is_diff=True)
    w_dwi = get_w(slc_dwi, slc_d_mask, is_diff=True)
    w_adc = get_w(slc_adc, slc_d_mask)

    format_ax(axes[0], "T1c (Fixed Anchor)", slc_t1c, vmin=w_t1c[0], vmax=w_t1c[1], mask_slc=slc_s_mask)
    format_ax(axes[1], "T1 (Mapped)", slc_t1, vmin=w_t1[0], vmax=w_t1[1], mask_slc=slc_s_mask)
    format_ax(axes[2], "T2 (Mapped)", slc_t2, vmin=w_t2[0], vmax=w_t2[1], mask_slc=slc_s_mask)

    rgb_s = np.zeros((*slc_t1c.shape, 3))
    rgb_s[..., 0] = norm_rgb(slc_t1c, slc_s_mask)
    rgb_s[..., 1] = norm_rgb(slc_t2, slc_s_mask)
    rgb_s[..., 2] = norm_rgb(slc_t2, slc_s_mask)
    format_ax(axes[3], "Struct Overlay (Red=T1c, Cyan=T2)", rgb_s, is_overlay=True)

    # Use diff mask (slc_d_mask) specifically here to prevent FOV gray boxes
    format_ax(axes[4], "B0 (Z-Scored)", slc_b0, vmin=w_b0[0], vmax=w_b0[1], mask_slc=slc_d_mask)
    format_ax(axes[5], "DWI (Z-Scored)", slc_dwi, vmin=w_dwi[0], vmax=w_dwi[1], mask_slc=slc_d_mask)
    format_ax(axes[6], "ADC (Unscaled Physics)", slc_adc, vmin=w_adc[0], vmax=w_adc[1], mask_slc=slc_d_mask)

    rgb_d = np.zeros((*slc_t2.shape, 3))
    rgb_d[..., 0] = norm_rgb(slc_t2, slc_s_mask)
    rgb_d[..., 1] = norm_rgb(slc_b0, slc_d_mask, is_diff=True)
    rgb_d[..., 2] = norm_rgb(slc_b0, slc_d_mask, is_diff=True)
    format_ax(axes[7], "Diff Overlay (Red=T2, Cyan=B0)", rgb_d, is_overlay=True)

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)

def save_multimodal_qa(t1c, t2, b0, adc, struct_mask, diff_mask, patient_id, output_path, num_slices=6):
    resampler_z = sitk.ResampleImageFilter()
    resampler_z.SetReferenceImage(t2)
    resampler_z.SetInterpolator(sitk.sitkLinear)
    resampler_z.SetDefaultPixelValue(0.0)
    b0_mapped = resampler_z.Execute(b0)
    adc_mapped = resampler_z.Execute(adc)

    resampler_mask = sitk.ResampleImageFilter()
    resampler_mask.SetReferenceImage(t2)
    resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_mask.SetDefaultPixelValue(0)
    diff_mask_mapped = resampler_mask.Execute(diff_mask)

    arr_t1c, arr_t2 = sitk.GetArrayViewFromImage(t1c), sitk.GetArrayViewFromImage(t2)
    arr_b0, arr_adc = sitk.GetArrayViewFromImage(b0_mapped), sitk.GetArrayViewFromImage(adc_mapped)
    arr_s_mask = sitk.GetArrayViewFromImage(struct_mask)
    arr_d_mask = sitk.GetArrayViewFromImage(diff_mask_mapped)

    z_valid = np.where(np.any(arr_b0 > 1e-5, axis=(1, 2)))[0]
    if len(z_valid) == 0: return
    z_min, z_max = z_valid[0], z_valid[-1]
    z_slices = np.linspace(z_min, z_max, num_slices, dtype=int)

    fig, axes = plt.subplots(5, num_slices, figsize=(3.5 * num_slices, 17.5))
    fig.patch.set_facecolor('black')
    fig.suptitle(f"{patient_id} | Multi-Slice Z-Axis Verification\nZ-Range: {z_min} to {z_max}", color="white", fontsize=18, fontweight='bold', y=0.96)

    def get_w(slc, mask_slc, is_diff=False):
        valid = slc[mask_slc > 0.5]
        if len(valid) == 0: return 0, 1
        p_low = np.percentile(valid, 2 if is_diff else 1)
        p_high = np.percentile(valid, 95 if is_diff else 99)
        return (0, 1) if p_high - p_low < 1e-8 else (p_low, p_high)

    def norm_rgb(slc, mask_slc, is_diff=False):
        valid = slc[mask_slc > 0.5]
        if len(valid) == 0: return np.zeros_like(slc)
        p1 = np.percentile(valid, 1)
        p99 = np.percentile(valid, 95 if is_diff else 99)
        if p99 - p1 < 1e-8: return np.zeros_like(slc)
        normed = np.clip((slc - p1) / (p99 - p1), 0, 1)
        normed[mask_slc < 0.5] = 0.0
        return normed

    for i, z in enumerate(z_slices):
        s_t1c, s_t2 = np.flipud(np.fliplr(arr_t1c[z])), np.flipud(np.fliplr(arr_t2[z]))
        s_b0, s_adc = np.flipud(np.fliplr(arr_b0[z])), np.flipud(np.fliplr(arr_adc[z]))
        s_s_mask = np.flipud(np.fliplr(arr_s_mask[z]))
        s_d_mask = np.flipud(np.fliplr(arr_d_mask[z]))

        wt1c, wt2 = get_w(s_t1c, s_s_mask), get_w(s_t2, s_s_mask)
        wb0, wadc = get_w(s_b0, s_d_mask, is_diff=True), get_w(s_adc, s_d_mask)

        disp_t1c = np.where(s_s_mask > 0.5, s_t1c, wt1c[0])
        disp_t2  = np.where(s_s_mask > 0.5, s_t2,  wt2[0])
        disp_b0  = np.where(s_d_mask > 0.5, s_b0,  wb0[0])
        disp_adc = np.where(s_d_mask > 0.5, s_adc, wadc[0])

        axes[0, i].imshow(disp_t1c, cmap='gray', vmin=wt1c[0], vmax=wt1c[1])
        if i == 0: axes[0, i].set_ylabel("Anchor\n(T1c)", color="white", fontsize=14, fontweight='bold')
        axes[0, i].set_title(f"Slice Z: {z}", color="white", fontsize=16, pad=10)

        axes[1, i].imshow(disp_t2, cmap='gray', vmin=wt2[0], vmax=wt2[1])
        if i == 0: axes[1, i].set_ylabel("Struct\n(T2)", color="white", fontsize=14, fontweight='bold')

        axes[2, i].imshow(disp_b0, cmap='gray', vmin=wb0[0], vmax=wb0[1])
        if i == 0: axes[2, i].set_ylabel("Diff\n(B0)", color="white", fontsize=14, fontweight='bold')

        axes[3, i].imshow(disp_adc, cmap='gray', vmin=wadc[0], vmax=wadc[1])
        if i == 0: axes[3, i].set_ylabel("Diff\n(ADC)", color="white", fontsize=14, fontweight='bold')

        rgb = np.zeros((*s_t2.shape, 3))
        rgb[..., 0] = norm_rgb(s_t2, s_s_mask)
        rgb[..., 1] = norm_rgb(s_b0, s_d_mask, is_diff=True)
        rgb[..., 2] = norm_rgb(s_b0, s_d_mask, is_diff=True)
        axes[4, i].imshow(rgb)
        if i == 0: axes[4, i].set_ylabel("Overlay\n(T2+B0)", color="white", fontsize=14, fontweight='bold')

        for row in range(5):
            axes[row, i].set_xticks([]); axes[row, i].set_yticks([])
            for spine in axes[row, i].spines.values(): spine.set_color('#1a1a1a')

    plt.subplots_adjust(top=0.92, wspace=0.05, hspace=0.05)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
