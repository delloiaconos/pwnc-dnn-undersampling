""" 
Article: Smart Water Meter based on Deep Neural Network and Under-Sampling for PWNC Detection
Authors: Marco Carratu, Salvatore Dello Iacono, Giuseppe Di Leo, Vincenzo Gallo, Consolatina Liguori and Antonio Pietrosanto

In case of doubt or questions contact: sdelloiacono[at]unisa.it or vgallo[at]unisa.it

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os, pickle
import cv2
import numpy as np


DATADIR  = "../data/"
OUTDIR   = "../train/"

CLASSES  = ["0","2","4","8"]
SEQ_SIZE = 3     # Number of Images per Sequence 
SEQ_N    = 187   # Number of Sequences per Class

IMG_ROWS = 100
IMG_COLS = 100

test_data = np.array([])


classesN = len( CLASSES )
training_set_size = SEQ_N*classesN # Image number


training_data = np.zeros( (training_set_size,SEQ_SIZE,IMG_COLS,IMG_ROWS,1) )
class_data    = np.zeros((training_set_size))

if __name__=='__main__':
    print("Ready...")

    j=0 
    for iclass in CLASSES: 
        print( f"Reading class '{iclass}'..." )

        path = os.path.join(DATADIR, iclass) 
        class_num = CLASSES.index(iclass) 
        k = 0 

        images = os.listdir(path)
        inum = 0
        while inum < SEQ_N*SEQ_SIZE:

            if k < SEQ_SIZE: 
                fname = os.path.join( path, images[inum]  )
                try:
                    print( f"\tReading image '{fname}'..." )
                    training_data[j,k,:,:,0] = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
                except Exception as e:
                    print(e,flush=True)
                    pass 
                k=k+1
            if k==SEQ_SIZE:
                class_data[j] = class_num
                k=0
                j=j+1
            inum = inum + 1
      
    print("Pickling...")

    with open( OUTDIR + "x.pickle" , 'wb') as handle:
        pickle.dump(training_data.astype(np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open( OUTDIR + "y.pickle" , 'wb') as handle:
        pickle.dump(class_data.astype(np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done!")