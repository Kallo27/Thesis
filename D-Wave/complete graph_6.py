from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s_1': 23/2, 's_2': 11, 's_3': 21/2, 's_4': 10, 's_5': 19/2, 's_6': 12}
J = {('s_1', 's_2'): 3, ('s_1', 's_3'): 3, ('s_1', 's_4'): 3, ('s_1', 's_5'): 3, ('s_1', 's_6'): 3, ('s_2', 's_3'): 3,  
    ('s_2', 's_4'): 3, ('s_2', 's_5'): 3, ('s_2', 's_6'): 3, ('s_3', 's_4'): 3, ('s_3', 's_5'): 3, ('s_3', 's_6'): 3,
    ('s_4', 's_5'): 3, ('s_4', 's_6'): 3, ('s_5', 's_6'): 3}

sampleset = sampler.sample_ising(h, J, num_reads=5000, label='knapsack problem - 5 objects')

print(sampleset)
print(sampleset.info)

dwave.inspector.show(sampleset)