import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("./tools/log.csv")

metric = 'dice_lesao'
train_metric = log['Train_'+metric].rolling(window=1).mean().tolist()
val_metric = log['Valid_'+metric].rolling(window=1).mean().tolist()

plt.plot(train_metric)
plt.plot(val_metric)
plt.title('Model {}'.format(metric))
plt.ylabel(metric)
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
