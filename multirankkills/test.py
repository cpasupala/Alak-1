import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

ylabel = []
X = []
if Path("./alak_y_addon.pkl").is_file():
    print("File alak_y.pkl exists")
    ylabel = pickle.load(open("./alak_y_addon.pkl","rb"))
    print(ylabel.shape,np.unique(ylabel),type(ylabel[0]))
if Path("./alak_x_addon.pkl").is_file():
    print("File alak_x.pkl exists")
    X = pickle.load(open("./alak_x_addon.pkl","rb"))
    print(X.shape,np.unique(X),type(X[0][0]))
else:
    print("File Does not exist")
if Path("./perfect_model.pkl").is_file():
    model = pickle.load(open("./perfect_model.pkl","rb"))
    print(type(model))

print(ylabel.mean())

#print(type(ylabel[0]))
#ylabel = ylabel.astype(int)
#print(type(ylabel[0]),np.unique(ylabel))
#print(type(X[0][0]),np.unique(X))
#X = np.where(X=='x','1',X)
#print(type(X[0][0]),np.unique(X))
#X = np.where(X=='_','0',X)
#print(type(X[0][0]),np.unique(X))
#X = np.where(X=='o','-1',X)
#print(type(X[0][0]),np.unique(X))
#X = X.astype(int)
#print(type(X[0][0]),np.unique(X))

print(len(np.array([])))
