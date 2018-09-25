import glob
import matplotlib.pyplot as plt
import numpy as np
import random

train_english_files = glob.glob('data/train/en/*.jpg')
train_french_files = glob.glob('data/train/fr/*.jpg')

print('english files:', len(train_english_files))
print('french files:', len(train_french_files))

def random_grab(img, image_edge_length):

    height, width = img.shape

    row_ind = random.randint(0, height - image_edge_length)
    col_ind = random.randint(0, width - image_edge_length)

    crop = img[row_ind : row_ind + image_edge_length, col_ind : col_ind + image_edge_length]

    return crop


#plt.figure()
#plt.imshow(image, cmap = 'Greys')

#print(np.mean(image))

z_means = []

# 520 and 526 are mislabelled

for i in range(500,600):

    image = train_english_files[i]
    image = plt.imread(image, format = 'jpeg')

    image = 1 - (image / float(np.max(image)))

    for j in range(50):


        z = random_grab(image, 300)

        z_means.append(np.mean(z))

        if np.mean(z) > 0.05:

            plt.figure()
            plt.imshow(z, cmap = 'Greys')
            plt.savefig('data/train/en_crops/' + str(i) + '_' + str(j) + '.jpg')
            plt.close()
        #    plt.figure()
        #    plt.imshow(z, cmap = 'Greys')
        #    plt.show()

plt.figure()
plt.hist(z_means, bins = 20)

plt.show()


'''
plt.figure()
plt.imshow(z, cmap = 'Greys')

plt.show()
'''
