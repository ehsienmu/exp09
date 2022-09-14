
import pickle
from matplotlib import pyplot as plt

# Load


with open('dct_var_record.pk', 'rb') as f:
    dct_var = pickle.load(f)
x = []
y = []
for k, v in dct_var.items():
    x.append(k)
    y.append(v.item())
    print(k, v)
plt.plot(x, y)
plt.savefig('./var.png')

# print(new_dict)