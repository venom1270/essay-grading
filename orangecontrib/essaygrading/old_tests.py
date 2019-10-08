import numpy as np

def gearysC():
    C = np.array([[1.5, 2, 1.5]])

    doc = np.array([[1, 1, 1], [0, 1, 2], [2, 2, 1], [3, 4, 2]])
    N = doc.shape[0]  # st. tock
    n = doc.shape[1]  # st. komponent
    S = (N - 1) * 2  # vsota utezi TODO: je to res: sosedov je n-1 ??
    c = 0  # sprotna vsota
    doc_i = 0
    D = doc - C[doc_i]
    # treba je narest v dveh delih, ker so notrnji >1 in <n elementi dvakrat v formuli (stevec), imenovalec je isti
    D = doc
    Di = np.delete(D, -1, axis=0)
    Dj = np.delete(D, 0, axis=0)
    D_numerator = np.sum(np.power(np.subtract(Di, Dj), 2), axis=0)
    print(Di)
    D_denominator = np.sum(np.power(Di - C[doc_i], 2), axis=0)
    Dj = np.delete(D, -1, axis=0)
    Di = np.delete(D, 0, axis=0)
    D_numerator += np.sum(np.power(np.subtract(Di, Dj), 2), axis=0)
    print(Di)
    print(C[doc_i])
    D_denominator += np.sum(np.power(Di - C[doc_i], 2), axis=0)
    print(D_numerator)
    print(D_denominator)
    c = np.sum(np.divide(D_numerator, D_denominator))

    if N < 2:
        c = 0
    else:
        c = ((N - 1) / 2) * (c / n)
    print(c)
    exit()