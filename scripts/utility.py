import numpy as np 

from matplotlib import pyplot

# constant 
FM_LABEL = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
DG_LABEL = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

#cpu_model = NN_test.CNN_model.sync_to_cpu()

def plot_predictions(images, predictions, true_labels, dataset):
    # set label data for plot later
    if dataset == 'mnist':
        LABEL_NAMES = DG_LABEL
    elif dataset == 'fashion_mnist':
        LABEL_NAMES = FM_LABEL
    
    n = images.shape[0]
    nc = int(np.ceil(n / 4))
    fig = pyplot.figure(figsize=(4,3))
    # axes = fig.add_subplot(nc, 4)
    f, axes = pyplot.subplots(nc, 4)
    f.tight_layout()
  
    for i in range(nc * 4):
        y = i // 4
        x = i % 4

        axes[x, y].axis('off')
    
        label = LABEL_NAMES[np.argmax(predictions[i])]
        confidence = np.max(predictions[i])
        
        if i > n:
            continue
        
        axes[x, y].imshow(images[i])
        pred_label = np.argmax(predictions[i])
        axes[x, y].set_title("{} ({})\n {:.3f}".format(
            LABEL_NAMES[pred_label], 
            LABEL_NAMES[true_labels[i]],
            confidence
        ), color=("green" if true_labels[i] == pred_label else "red"))

    pyplot.gcf().set_size_inches(8, 8)
