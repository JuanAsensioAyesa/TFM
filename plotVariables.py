import numpy 
import matplotlib.pyplot as plt

def leerLista(filename):
    L = []
    f = open(filename)
    for data in f:
        L.append(data)
    f.close()
    return L

def plotList(list,show = True):
    plt.plot(list)
    plt.grid()
    if show:
        plt.show()



if __name__ == "__main__":
    print("hola")