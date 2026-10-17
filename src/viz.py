import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skimage import segmentation, color

def overlay(img, lab, voids, edge_ids):
    rgb = np.dstack([img]*3)
    b = segmentation.find_boundaries(lab, mode='inner')
    edge_mask = np.isin(lab, list(edge_ids)) if edge_ids else np.zeros_like(b)
    rgb[b & ~edge_mask] = (1, 0.15, 0.15)
    rgb[b & edge_mask] = (1, 0.85, 0.1)
    rgb[voids] = rgb[voids]*0.4 + np.array([0, 0.4, 1])*0.6
    return rgb

def panel(name, img, out, key, path, title_extra=''):
    M = out[key]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = ax.ravel()
    ax[0].imshow(img, cmap='gray'); ax[0].set_title('1. Original')
    ax[1].imshow(out['pre'], cmap='gray'); ax[1].set_title('2. Preprocessed')
    ax[2].imshow(M['edge'], cmap='magma'); ax[2].set_title('3. Probability map (RF)' if key=='C' else '3. Edge / boundary map')
    raw = color.label2rgb(M['labels'], bg_label=0, alpha=1) if True else None
    ax[3].imshow(segmentation.mark_boundaries(img, M['labels'], color=(1,0,0))); ax[3].set_title(f"4. Raw segmentation ({M['labels'].max()} obj.)")
    ax[4].imshow(color.label2rgb(M['post'], bg_label=0, alpha=1)); 
    n_in = len([g for g in M['grains'] if not g['touches_border']])
    ax[4].set_title(f"5. Postprocessed ({M['post'].max()} grains, {n_in} inner)")
    ax[5].imshow(overlay(img, M['post'], M['voids'], M['edge_ids'])); ax[5].set_title('6. Overlay (red=grain, yellow=edge, blue=void)')
    for a in ax: a.axis('off')
    fig.suptitle(f"{name} — method {key}  L={out['params']['L']:.1f}px {title_extra}")
    plt.tight_layout(); plt.savefig(path, dpi=70); plt.close()
