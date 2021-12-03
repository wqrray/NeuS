# Plot the results
import tensorflow as tf
import matplotlib.pyplot as plt

color_losses = []
eikonal_losses = []
losses = []
psnrs = []

path_to_events_file = "events.out.tfevents.1638464843.featurize.19218.0"
try:
    for e in tf.train.summary_iterator(path_to_events_file):
        for v in e.summary.value:
            if "color_loss" in v.tag:
                color_losses.append(v.simple_value)
            if "eikonal_loss" in v.tag:
                eikonal_losses.append(v.simple_value)
            if "Loss/loss" in v.tag:
                losses.append(v.simple_value)
            if "psnr" in v.tag:
                psnrs.append(v.simple_value)
except:
    pass

plt.figure()
plt.plot(color_losses)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('color losses')

plt.figure()
plt.plot(eikonal_losses)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('eikonal losses')

plt.figure()
plt.plot(losses)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('total losses')

plt.figure()
plt.plot(psnrs)
plt.xlabel('iteration')
plt.ylabel('psnr')
plt.title('psnrs')

plt.show()