import pickle
import pandas as pd
import os
import torch
from matplotlib import pyplot as plt
import seaborn as sns

# Load
with open('attack_record.pk', 'rb') as f:
    new_dict = pickle.load(f)


# print(new_dict.items())

result_list = []
train_acc = []

for k, i in new_dict.items():
    result_list.append((k,)+(i))

for i in range(len(new_dict)):
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    ckpt_path = './checkpoint/sf_resnet_freq'+str(i)+'.pth'
    checkpoint = torch.load(ckpt_path)
    best_acc = checkpoint['acc']
    # print(best_acc)
    # print(best_acc.item())
    train_acc.append((best_acc.item()))
    # dd

df = pd.DataFrame(result_list, columns=['dct_ind',  'val_acc', 'Attack_Success_Rate', 'Test_Accuracy:', 'Attacked_Accuracy'])
df['val_acc'] = train_acc

# print(df.drop('val_acc', axis=1))


# df.drop(['val_acc', 'dct ind'], axis=1).plot()
# plt.title('attack')
# plt.savefig('./result.png')

# print(new_dict)
df.drop(['val_acc', 'Test_Accuracy:'], axis=1, inplace=True)
with open('../dct_var_record.pk', 'rb') as f:
    Frequency_Variance = pickle.load(f)
freq_ind  = []
freq_var = []
for k, v in Frequency_Variance.items():
    freq_ind.append(k)
    freq_var.append(v.item())
plt.title('freq to Frequency_Variance')
# plt.savefig('./Frequency_Variance.png')
df['Frequency_Variance'] = freq_var[:len(df)]
# print(new_dict)



sns.set()


df = df.set_index('dct_ind')

fig = plt.figure(figsize=(20,10)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .3

df.Attacked_Accuracy.plot(kind='bar',color='red',ax=ax,width=width*2, position=0)
df.Attack_Success_Rate.plot(kind='bar',color='green',ax=ax,width=width, position=0)
df.Frequency_Variance.plot(kind='bar',color='blue', ax=ax2,width = width,position=1)

# ax.grid(None, axis=1)
# ax2.grid(None)

ax.legend(loc='right', bbox_to_anchor=(1.0, 1.05))
ax2.legend(loc='right', bbox_to_anchor=(1.0, 1.11))
# ax.legend(loc='upper right', bbox_to_anchor=(1.05, 0.5))

# plt.legend(bbox_to_anchor=(1.05, 1.05), loc=2, borderaxespad=0.)

# ax.legend(loc='upper right', bbox_to_anchor=(1.05, 0.5))
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_ylabel('Accuracy (for red and green)')
ax2.set_ylabel('Variance (for blue)')
ax.set_xlabel('Frequency')
# ax.set_xlim(-1,7)


plt.savefig('./final.png')