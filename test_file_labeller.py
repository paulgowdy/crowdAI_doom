import glob
import matplotlib.pyplot as plt

test_files = glob.glob('data/test_images/*.jpg')

print(len(test_files))


plt.figure(figsize = (8,8))
for i in range(400,500):

    print(i, test_files[i])
    image = plt.imread(test_files[i], format = 'jpeg')

    width = image.shape[1]
    height = image.shape[0]

    image = image[:int(height/2.0), :int(width/2.0)]

    #print(image.shape)

    plt.clf()
    plt.imshow(image, cmap = 'Greys')
    plt.pause(0.1)

    label = 0

    while label not in [1, 2]:

        label = int(input('>>'))

        if label == 1:

            with open('data/english_test_files.txt', 'a') as f:

                f.write(test_files[i])
                f.write('\n')

        if label == 2:

            with open('data/french_test_files.txt', 'a') as f:

                f.write(test_files[i])
                f.write('\n')
