import matplotlib.pyplot as plt
import numpy as np
import os
import skimage

import time

import joblib
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

#Modles

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

# Printing confution matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""   
    Parameter
    ---------
    src: string path to data
    pklname: string path to output file
    width: int: target width of the image in pixels
    height: height of image.
    include: set[str] set containing str
"""

#Used to read data from a folder. Outputs a binary file so this only needs to be run once
def Get_croped_img_data(src, pklname, include, width=3000, height=3000):

     
    data = dict()
    data['description'] = 'resized ({0}x{1})MTG sets in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)
 
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))                    
                    im= im[1655:1830, 2220:2385] #This is ~there the set symbol is located
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)
            

        joblib.dump(data, pklname)

#IMPORTANT you need to change this to the path with card set
data_path = 'D:\Vis Studio C code workshop\Python\Ml project\Card Sets'

#Should print a list of all sets if set correctly 
print(os.listdir(data_path)) 
width=3000
base_name = 'Set Names'
#set names
include = {'Dominaria', 'Ice Age', 'Innistrad','Mirage','Theros','Unstable'}

#gets the data
start = time.time()
Get_croped_img_data(src=data_path, pklname=base_name, width=width, include=include)
end = time.time()
print('Time taken for Get_croped_img_data: ', end - start)

#If there is a binary file then load it else...
data = joblib.load(f'{base_name}_{width}x{width}px.pkl')

#Simple error check
# print('number of samples: ', len(data['data']))  =     112
# print('keys: ', list(data.keys()))   keys: = ['description', 'label', 'filename', 'data']
# print('description: ', data['description']) =description:  resized (3000x3000)MTG sets in rgb
# print('image shape: ', data['data'][0].shape) =(175, 165, 3)
# print('labels:', np.unique(data['label'])) =['Dominaria' 'Ice Age' 'Innistrad' 'Mirage' 'Theros' 'Unstable']
 
# use np.unique to get all unique values in the list of labels
labels = np.unique(data['label'])


X = np.array(data['data'])
y = np.array(data['label'])

 
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,
    shuffle=True,
    random_state=42,
)


#This class turns the data into a gray image
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])


#This function plots the confution matrix
def plot_confusion_matrix(cmx, vmax1=None, vmax2=None, vmax3=None):
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    cmx_zero_diag = cmx_norm.copy()
 
    np.fill_diagonal(cmx_zero_diag, 0)
 
    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(12, 3)
    [a.set_xticks(range(len(cmx)+1)) for a in ax]
    [a.set_yticks(range(len(cmx)+1)) for a in ax]
         
    im1 = ax[0].imshow(cmx, vmax=vmax1)
    ax[0].set_title('as is')
    im2 = ax[1].imshow(cmx_norm, vmax=vmax2)
    ax[1].set_title('%')
    im3 = ax[2].imshow(cmx_zero_diag, vmax=vmax3)
    ax[2].set_title('% and 0 diagonal')
 
    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) 
                        for divider in dividers]
 
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.tight_layout()
    plt.show()


grayify = RGB2GrayTransformer()
scalify = StandardScaler()

#print(X_train.shape)    #===(89, 175, 165, 3) 
X_train_gray = grayify.fit_transform(X_train)

nsamples, nx, ny = X_train_gray.shape

X_train_grey_reshaped2D = X_train_gray.reshape((nsamples,nx*ny)) #needs to be a 2d array

X_train_prepared = scalify.fit_transform(X_train_grey_reshaped2D)

X_test_grey = grayify.fit_transform(X_test)
nsamples, nx, ny = X_test_grey.shape
X_test_grey_reshaped2D = X_test_grey.reshape((nsamples,nx*ny))
X_test_prepared = scalify.fit_transform(X_test_grey_reshaped2D)
#print(X_train_prepared.shape)              (89, 28875)

#Start SGDClassifier
start = time.time()
sgd_clf = SGDClassifier(random_state=42, max_iter=1000)
sgd_clf.fit(X_train_prepared, y_train)
end = time.time()
print('Time taken for SGDClassifier: ', end - start)
y_pred_test = sgd_clf.predict(X_test_prepared)

print('Percentage correct: ', 100*np.sum(y_pred_test == y_test)/len(y_test))
cmx = confusion_matrix(y_test, y_pred_test)
print(cmx)
plot_confusion_matrix(cmx)
#End of SGDClassifier


#Start K Neighbors Classifier
start = time.time()
sgd_knerest = KNeighborsClassifier(n_neighbors=3)
sgd_knerest.fit(X_train_prepared, y_train)
end = time.time()
print('Time taken for KNeighborsClassifier: ', end - start)


y_pred_test = sgd_knerest.predict(X_test_prepared)

print('Percentage correct: ', 100*np.sum(y_pred_test == y_test)/len(y_test))
cmx = confusion_matrix(y_test, y_pred_test)
print(cmx)
plot_confusion_matrix(cmx)
print('\n', sorted(np.unique(y_test)))
#End of K Neighbors Classifier


#Start of Logistic Regression
start = time.time()

sgd_logR=LogisticRegression(random_state=42,max_iter=1000, tol=1e-3)
sgd_logR.fit(X_train_prepared, y_train)
end = time.time()
print('Time taken for LogisticRegression: ', end - start)
y_pred_test = sgd_logR.predict(X_test_prepared)

print('Percentage correct: ', 100*np.sum(y_pred_test == y_test)/len(y_test))

 
cmx = confusion_matrix(y_test, y_pred_test)
print(cmx)
plot_confusion_matrix(cmx)
print('\n', sorted(np.unique(y_test)))

#End of Logistic Regression

     


