from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s_1': 47/2, 's_2': 23, 's_3': 45/2, 's_4': 22, 's_5': 43/2, 's_6': 21, 's_7': 41/2, 's_8': 24}
J = {('s_1', 's_2'): 4.0, ('s_1', 's_3'): 4.0, ('s_1', 's_4'): 4.0, ('s_1', 's_5'): 4.0, ('s_1', 's_6'): 4.0, ('s_1', 's_7'): 4.0,
    ('s_1', 's_8'): 4.0, ('s_2', 's_3'): 4.0, ('s_2', 's_4'): 4.0, ('s_2', 's_5'): 4.0, ('s_2', 's_6'): 4.0, ('s_2', 's_7'): 4.0, 
    ('s_2', 's_8'): 4.0, ('s_3', 's_4'): 4.0, ('s_3', 's_5'): 4.0, ('s_3', 's_6'): 4.0, ('s_3', 's_7'): 4.0, ('s_3', 's_8'): 4.0, 
    ('s_4', 's_5'): 4.0, ('s_4', 's_6'): 4.0, ('s_4', 's_7'): 4.0, ('s_4', 's_8'): 4.0, ('s_5', 's_6'): 4.0, ('s_5', 's_7'): 4.0, 
    ('s_5', 's_8'): 4.0, ('s_6', 's_7'): 4.0, ('s_6', 's_8'): 4.0, ('s_7', 's_8'): 4.0}

sampleset = sampler.sample_ising(h, J, num_reads=5000, label='knapsack problem - 7 objects')

print(sampleset)
print(sampleset.info)

dwave.inspector.show(sampleset)