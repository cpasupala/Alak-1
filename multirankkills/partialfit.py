import pickle
from pathlib import Path
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    savemodel = True
    if Path("./alak_y_addon.pkl").is_file():
        y = pickle.load(open("./alak_y_addon.pkl","rb"))
    if Path("./alak_x_addon.pkl").is_file():
        X = pickle.load(open("./alak_x_addon.pkl","rb"))

    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

    print(f'Training data size: {x_train.shape}, {y_train.shape}\nUniques:{np.unique(x_train)},{np.unique(y_train)}')
    print(f'Test data size: {x_test.shape}, {y_test.shape}')

    if Path("./perfect_model.pkl").is_file():
        model = pickle.load(open("./perfect_model.pkl","rb"))
    else:
        raise Exception("Could not find the model. Exiting")

    model.partial_fit(x_train,y_train)
    print(f'Score on Trained Data: {model.score(x_train,y_train):.2f}')
    print(f'Score on Testing Data: {model.score(x_test,y_test):.2f}')


    if Path("./alak_y.pkl").is_file():
        y_old = pickle.load(open("./alak_y.pkl","rb"))
    if Path("./alak_x.pkl").is_file():
        X_old = pickle.load(open("./alak_x.pkl","rb"))
    print(f'Score on previous Data: {model.score(X_old,y_old):.2f}')

    if(savemodel):
        with open('perfect_model_improved.pkl','wb') as f:
            pickle.dump(model,f)
