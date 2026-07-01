import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

wc_color_dict = {
    0: (0,0,0),
    10: (0,100,0),
    20: (255,187,34),
    30: (255,255,76),
    40: (240,150,255),
    50: (250,0,0),
    60: (180,180,180),
    70: (240,240,240),
    80: (0,100,200),
    90: (0,150,160),
    95: (0,207,117),
    100: (250,230,160)
}





def sentinel_worldcover_image_and_mask_display(data, mask, label_map = None, wc_code_map = None, alpha = 0.25):
    
    # Set the color map for the display
    color_dict ={}
    if wc_code_map == None:
        for i in sorted(np.unique(mask)):
            color_dict[i] = (255 * random(), 255 * random(), 255 * random())
    else:
        for key, value in wc_code_map.items():
            color_dict[key] = wc_color_dict[value]


    # Setup the Legend
    if label_map != None:
        legend_elements = [
            Patch(facecolor=tuple(np.array(color_dict[i])/255), label=label_map[i])
            for i in sorted(color_dict)
        ]
    else:
        legend_elements = [
            Patch(facecolor=tuple(np.array(color_dict[i])/255), label=i)
            for i in sorted(color_dict)
        ]


    # Prep and normalize input
    rgb = data[[2,1,0], :, :]
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb.astype(np.float32)
    rgb = 5* ((rgb - rgb.min()) / (rgb.max() - rgb.min()))

    # Prep mask to overlay
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in color_dict.items():
        rgb_mask[mask == label] = color


    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.imshow(rgb_mask, alpha=alpha)
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    plt.axis("off")
    plt.show()




def display_change_mask_on_image(data, mask, alpha = 1):
    '''
    Expects mask to be a change mask with two values
        1's are displayed as red
        -1's are displayed as green
    '''

    # Prep and normalize input
    rgb = data[:, :, [2,1,0]]
    rgb = rgb.astype(np.float32)
    rgb = 5* ((rgb - rgb.min()) / (rgb.max() - rgb.min()))


    # Prep mask overlay
    rgb_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgb_mask[mask == 1] = [255, 0, 0, 255]      # loss
    rgb_mask[mask == -1] = [0, 255, 0, 255]     # growth

    #alpha_mask = np.zeros(mask.shape)
    #alpha_mask[mask == 1] = alpha
    #alpha_mask[mask == -1] = alpha

    legend_elements = [
        Patch(facecolor='red', label='Vegetation Loss'),
        Patch(facecolor='green', label='Vegetation Growth')
    ]

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.imshow(rgb_mask)
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    plt.axis("off")
    plt.show()
