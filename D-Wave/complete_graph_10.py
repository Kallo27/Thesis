import pandas as pd
from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s1': 79/2, 's2': 39, 's3': 77/2, 's4': 38, 's5': 75/2, 's6': 37, 's7': 73/2, 's8': 35, 's9': 71/2, 's10': 40}

J = {('s1', 's2'): 5, ('s1', 's3'): 5, ('s1', 's4'): 5, ('s1', 's5'): 5, ('s1', 's6'): 5, 
    ('s1', 's7'): 5, ('s1', 's8'): 5, ('s1', 's9'): 5, ('s1', 's10'): 5, ('s2', 's3'): 5, 
    ('s2', 's4'): 5, ('s2', 's5'): 5, ('s2', 's6'): 5, ('s2', 's7'): 5, ('s2', 's8'): 5, 
    ('s2', 's9'): 5, ('s2', 's10'): 5, ('s3', 's4'): 5, ('s3', 's5'): 5, ('s3', 's6'): 5, 
    ('s3', 's7'): 5, ('s3', 's8'): 5, ('s3', 's9'): 5, ('s3', 's10'): 5, ('s4', 's5'): 5, 
    ('s4', 's6'): 5, ('s4', 's7'): 5, ('s4', 's8'): 5, ('s4', 's9'): 5, ('s4', 's10'): 5, 
    ('s5', 's6'): 5, ('s5', 's7'): 5, ('s5', 's8'): 5, ('s5', 's9'): 5, ('s5', 's10'): 5,
    ('s6', 's7'): 5, ('s6', 's8'): 5, ('s6', 's9'): 5, ('s6', 's10'): 5, ('s7', 's8'): 5, 
    ('s7', 's9'): 5, ('s7', 's10'): 5, ('s8', 's9'): 5, ('s8', 's10'): 5, ('s9', 's10'): 5}


sampleset = sampler.sample_ising(h, J, num_reads=5000, label='complete graph - 10 qubits')

data = pd.DataFrame(sampleset)
data.to_excel('sampleset_10.xlsx', index=False)

print(sampleset)
print(sampleset.info)
dwave.inspector.show(sampleset)