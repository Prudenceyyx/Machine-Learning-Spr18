# import matplotlib.pyplot as plt

# import numpy as np
from collections import Counter


# x = np.linspace(0, 2 * np.pi, 400)
# y = np.sin(x ** 2)

folder = 'percept_data/'
# plt.close('all')
# f,axarr = plt.subplots(2,sharex=True)
# axarr[0].plot(x,y)
# axarr[1].scatter(x,y)
# plt.show()

# filename_train = "percept_data/" + 'spam_train.txt'
# with open(filename_train) as f:
#     emails = f.readlines()[:4000]
#     f.close()

# alpha = {}

# for email in emails:
#     words = Counter(email.strip().split()[1:])
#     for word in words:
#         alpha[word] = alpha.get(word, 0) + 1
# valid = [word for word,value in alpha.items() if value>25]
# print(valid)

with open(folder+"word.txt",'r') as f:
	content = f.readlines()
	print(len(content))

with open("weights.txt",'r') as f:
	content = f.read()
	print(len(content.split(',')))

