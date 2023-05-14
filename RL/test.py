import numpy as np
A = [1,2,3]
B = [4,5,6]
print(np.array(np.meshgrid(A,B)).T.reshape(-1,2))

inpdict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,'b':11,'c':12,'d':13}

while(True):
    x = input("enter a number")
    try: 
        print(f'{x} : {inpdict[x]}')
    except:
        print("Invalid key")
        continue

