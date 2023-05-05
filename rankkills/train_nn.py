import pickle
from pathlib import Path
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def encode(X,y):
    X = np.where(X=='x','1',X)
    X = np.where(X=='_','0',X)
    X = np.where(X=='o','-1',X)
    return X.astype(int),y.astype(int)

if __name__ == "__main__":
    savemodel= False
    if Path("./alak_y.pkl").is_file():
        y = pickle.load(open("./alak_y.pkl","rb"))
    if Path("./alak_x.pkl").is_file():
        X = pickle.load(open("./alak_x.pkl","rb"))

    #X, y = encode(X,y)
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

    print(f'Training data size: {x_train.shape}, {y_train.shape}\nUniques:{np.unique(x_train)},{np.unique(y_train)}')
    print(f'Test data size: {x_test.shape}, {y_test.shape}')

    clf = MLPClassifier(solver="adam", hidden_layer_sizes=[60,30], activation="relu", max_iter=int(1e7),learning_rate_init=float(5e-4))
    clf.fit(x_train,y_train)
    print(f'Score on Trained Data: {clf.score(x_train,y_train):.2f}')
    print(f'Score on Testing Data: {clf.score(x_test,y_test):.2f}')
    if(savemodel):
        with open('perfect_model.pkl','wb') as f:
            pickle.dump(clf,f)
