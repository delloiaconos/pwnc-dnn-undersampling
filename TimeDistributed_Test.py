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

from email.mime import image
import numpy as np
import tensorflow as tf
import os, csv
import seaborn as sns
import cv2

from DatasetCreator import CLASSES, SEQ_SIZE, IMG_ROWS, IMG_COLS


IOVERLAP = 0
DATADIR   = "../data/"
MODELFILE = "../models/model.h5"

# Check for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print( gpus )

model = tf.keras.models.load_model( MODELFILE )
print( model )

if __name__=='__main__':
    print("Starting Testing...")

    images = os.listdir(DATADIR)
    print( images )

    inum = 0
    k = 0
    test_vector = np.zeros( (1,SEQ_SIZE,IMG_COLS,IMG_ROWS,1) )
    out_vect = list()

    while inum < len( images ):

        if k < SEQ_SIZE: #Numero immagini per sequenza
            #fname = os.path.join( DATADIR, f"{images[inum]}.jpg" )
            fname = os.path.join( DATADIR, f"{images[inum]}" )
            try:
                print( f"\tReading image '{fname}'..." )
                
                img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
                test_vector[0,k,:,:,0] = cv2.resize(img, (IMG_ROWS, IMG_COLS)) 
            except Exception as e:
                print(e,flush=True)
                pass 
            k=k+1
        if k == SEQ_SIZE:
            # Start Testing on current sequence
            predict_x=model.predict(test_vector) 
            y_pred=np.argmax(predict_x,axis=1)
            out_vect.append( { 'time' : images[inum], 'class' : CLASSES[y_pred[0]] } )
            k=0
            inum = inum - IOVERLAP

        inum = inum + 1 
        
    j=0 #Seq_count
    
    print("Done!")

with open( '../results/' + 'results.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['time','class'])
    writer.writeheader()
    writer.writerows(out_vect)