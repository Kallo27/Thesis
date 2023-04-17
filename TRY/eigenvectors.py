import numpy as np
import pandas as pd

E = ([[1, -5, 8, 7],
    	[1, -2, 1, 6],
    	[2, -1, -5, 3],
    	[2, 7, -3, 5]])

EigValues, EigVectors = np.linalg.eig(E)
permute = EigValues.argsort()
EigValues = EigValues[permute]
EigVectors = EigVectors[:,permute]
print(EigVectors)

# Create a dataframe
data = {
	"eigenvector": ['e0', 'e1', 'e3', 'e4'],
	"value": [EigVectors[0], EigVectors[1], EigVectors[2], EigVectors[3]]
}

df2 = pd.DataFrame(EigVectors, columns=['e0', 'e1', 'e2', 'e3'])

df = pd.DataFrame(data)

# Save the dataframe to an Excel file
df2.to_excel('output.xlsx', index=False)
