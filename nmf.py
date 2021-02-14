import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

V = np.array([[0,2,0,0,0,1],
              [1,0,0,0,0,1],
              [0,0,3,1,0,0],
              [0,0,0,0,2,0],
              [0,0,0,1,0,0]])
              
V = pd.DataFrame(V, columns=['Lucia', 'Olga', 'Claudia', 'Miguel', 'Daniel', 'Sonia'])
V.index = ['Labial', 'Serum', 'Shampoo', 'Jabon', 'Colonia']

nmf = NMF(3)
nmf.fit(V)

H = pd.DataFrame(np.round(nmf.components_,2), columns=V.columns)
H.index = ['Una', 'Homem', 'Chronos']

W = pd.DataFrame(np.round(nmf.transform(V),2), columns=H.index)
W.index = V.index

reconstructed = pd.DataFrame(np.round(np.dot(W,H),2), columns=V.columns)
reconstructed.index = V.index

print ("",V)
