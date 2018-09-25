import glob
import matplotlib.pyplot as plt
import numpy as np

train_english_files = glob.glob('data/train/en/*.jpg')
train_french_files = glob.glob('data/train/fr/*.jpg')

print('english files:', len(train_english_files))
print('french files:', len(train_french_files))

en_means = []
fr_means = []

for i in range(400):#len(train_english_files)):

    image = plt.imread(train_english_files[i], format = 'jpeg')

    en_means.append(np.mean(image))

for i in range(400):#len(train_french_files)):

    image = plt.imread(train_french_files[i], format = 'jpeg')

    fr_means.append(np.mean(image))

print(len(en_means), len(fr_means))

plt.figure()
plt.hist(en_means, alpha = 0.3, bins = 30)
plt.hist(fr_means, alpha = 0.3, bins = 30)
plt.show()

'''
print(image.shape)
print(type(image))

plt.figure()
plt.imshow(image, cmap = 'Greys')
plt.show()
'''
