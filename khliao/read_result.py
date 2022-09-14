import pickle
import pandas as pd
import os
import torch
from matplotlib import pyplot as plt

# Load
with open('attack_record.pk', 'rb') as f:
    new_dict = pickle.load(f)


# print(new_dict.items())

result_list = []
train_acc = []

for k, i in new_dict.items():
    result_list.append((k,)+(i))

for i in range(53):
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    ckpt_path = './checkpoint/sf_resnet_freq'+str(i)+'.pth'
    checkpoint = torch.load(ckpt_path)
    best_acc = checkpoint['acc']
    # print(best_acc)
    # print(best_acc.item())
    train_acc.append((best_acc.item()))
    # dd

df = pd.DataFrame(result_list, columns=['dct ind',  'val_acc', 'Attack Success Rate', 'Test Accuracy:', 'Attacked Accuracy'])
df['val_acc'] = train_acc

print(df.drop('val_acc', axis=1))


df.drop(['val_acc', 'dct ind'], axis=1).plot()
plt.savefig('./result.png')

# print(new_dict)