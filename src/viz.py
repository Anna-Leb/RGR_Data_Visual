import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skimage import segmentation, color
METHOD_NAMES = {'A': 'метод A — по границам', 'B': 'метод B — по яркости', 'C': 'метод C — обучаемая модель', 'E': 'метод E — общая модель'}

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
    ax[0].imshow(img, cmap='gray'); ax[0].set_title('1. Оригинал')
    ax[1].imshow(out['pre'], cmap='gray'); ax[1].set_title('2. Предобработка')
    ax[2].imshow(M['edge'], cmap='magma'); ax[2].set_title('3. Карта вероятностей границ' if key=='C' else '3. Карта границ')
    ax[3].imshow(segmentation.mark_boundaries(img, M['labels'], color=(1,0,0))); ax[3].set_title(f"4. Первичная сегментация ({M['labels'].max()} объектов)")
    ax[4].imshow(color.label2rgb(M['post'], bg_label=0, alpha=1)); 
    n_in = len([g for g in M['grains'] if not g['touches_border']])
    ax[4].set_title(f"5. Постобработка ({M['post'].max()} зёрен, целых {n_in})")
    ax[5].imshow(overlay(img, M['post'], M['voids'], M['edge_ids'])); ax[5].set_title('6. Наложение (красный — зерно, жёлтый — краевое, синий — пора)')
    for a in ax: a.axis('off')
    fig.suptitle(f"{name} — {METHOD_NAMES.get(key, key)} {title_extra}")
    plt.tight_layout(); plt.savefig(path, dpi=70); plt.close()
