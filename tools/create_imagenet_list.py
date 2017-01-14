import os

f = open('data/train.txt', 'w')
imagenet_basepath = './data/imagenet/'
for p1 in os.listdir(imagenet_basepath):
  if p1.startswith('n'):
    for p2 in os.listdir(imagenet_basepath + p1 + '/'):
      image = os.path.abspath(imagenet_basepath + p1 + '/' + p2)
      f.write(image + '\n')
f.close()
