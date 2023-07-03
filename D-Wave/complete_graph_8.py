from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s_1': 17, 's_2': 33/2, 's_3': 16, 's_4': 31/2, 's_5': 15, 's_6': 29/2, 's_7': 35/2}
J = {('s_1', 's_2'): 7/2, ('s_1', 's_3'): 7/2, ('s_1', 's_4'): 7/2, ('s_1', 's_5'): 7/2, ('s_1', 's_6'): 7/2, ('s_1', 's_7'): 7/2, 
    ('s_2', 's_3'): 7/2, ('s_2', 's_4'): 7/2, ('s_2', 's_5'): 7/2, ('s_2', 's_6'): 7/2, ('s_2', 's_7'): 7/2, ('s_3', 's_4'): 7/2, 
    ('s_3', 's_5'): 7/2, ('s_3', 's_6'): 7/2, ('s_3', 's_7'): 7/2, ('s_4', 's_5'): 7/2, ('s_4', 's_6'): 7/2, ('s_4', 's_7'): 7/2,
    ('s_5', 's_6'): 7/2, ('s_5', 's_7'): 7/2, ('s_6', 's_7'): 7/2}

sampleset = sampler.sample_ising(h, J, num_reads=5000, label='knapsack problem - 6 objects')

print(sampleset)
print(sampleset.info)

dwave.inspector.show(sampleset)