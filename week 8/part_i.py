########## part i) ########## 
def convol2d(kernel, inp_array, stride):
  k = kernel.shape
  n = inp_array.shape
  output = np.empty(shape=((n[0]-k[0])//stride + 1,(n[1]-k[1])//stride + 1))
  #output = np.empty(shape=(n[0]-stride-1,n[1]-stride-1))
  for i in range(output.shape[0]):
    for j in range(output.shape[1]):
      submatrix = inp_array[i:i+k[0],j:j+k[1]]
      output[i][j] = np.sum(np.multiply(submatrix, kernel))

  return output

##########  i) a) ########## 

n = int(input('Enter value of n'))
stride = int(input('Enter value for stride'))

import numpy as np
from numpy import random
inp_array = random.randint(20, size=(n,n))
# inp_array = np.array([[1, 2, 3, 4, 5],
#                       [1, 3, 2, 3, 10],
#                       [3, 2, 1, 4, 5],
#                       [6, 1, 1, 2, 2],
#                       [3, 2, 1, 5, 4]])
kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

print("Input matrix ", inp_array)
print("Kernel ",kernel)
output = convol2d(kernel, inp_array, stride)
print("Result of convolution ", output)

##########  i) b) ########## 
import numpy as np
from PIL import Image
im = Image.open('download1.png')
rgb = np.array(im.convert('RGB'))
r=rgb [ : , : , 0 ] # a r r a y o f R p i x e l s

img = Image.fromarray(np.uint8(r))
img.save('input.png')
kernel1 = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
kernel2 = np.array([[0, -1, 0],[-1, 8, -1],[0, -1, 0]])

output1 = convol2d(kernel1, r, 1)
print("Output with Kernel 1", output1)
img = Image.fromarray(np.uint8(output1))
img.save('output1.png')
img.show()

output2 = convol2d(kernel2, r, 1)
print("Output with Kernel 2", output2)

img = Image.fromarray(np.uint8(output2))
img.save('output2.png')
img.show()