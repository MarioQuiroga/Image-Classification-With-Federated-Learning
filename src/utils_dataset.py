import tensorflow as tf
import tensorflow_federated as tff
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as img

def unzipDataSet():
    zip_files = ['test1', 'train']

    for zip_file in zip_files:
        with zipfile.ZipFile("../data/{}.zip".format(zip_file),"r") as z:
            z.extractall("../data")
            print("{} unzipped".format(zip_file))
    TRAIN_DIR_PATH = '../data/train'
    file_names = os.listdir(TRAIN_DIR_PATH)
    print('There are {} number of images in directory.'.format(len(file_names)))

    train_dir='../data/train'
    if  not os.path.exists((os.path.join(train_dir,'cat'))):
        os.mkdir(os.path.join(train_dir,'cat'))
    if  not os.path.exists((os.path.join(train_dir,'dog'))):
        os.mkdir(os.path.join(train_dir,'dog'))
    for file in os.listdir(train_dir):
        if file[-3]=='j':
            if file[0]=='c':
                os.replace(os.path.join(train_dir,file),os.path.join(train_dir,'cat',file))
            else:
                os.replace(os.path.join(train_dir,file),os.path.join(train_dir,'dog',file))


BATCH_SIZE=32
IMAGE_SIZE=(128,128)
train_dir = '../data/train'
test_dir = '../data/test1'

def getDataTrain():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        validation_split=0.2,
        subset="training",
        seed=0
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=0
    )
    return train_ds, val_ds

# if __name__ == "__main__":
#     #unzipDataSet()
#     train_ds, val_ds = getDataTrain()
#     class_names = train_ds.class_names

#     plt.figure(figsize=(10, 10))
#     for images, labels in train_ds.take(1):
#         for i in range(9):
#             ax = plt.subplot(3, 3, i + 1)
#             plt.imshow(images[i].numpy().astype("uint8"))
#             plt.title(class_names[tf.argmax(labels[i])])
#             plt.axis("off")
#             plt.show()
