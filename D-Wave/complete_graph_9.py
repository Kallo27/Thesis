from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s_1': 31, 's_2': 61/2, 's_3': 30, 's_4': 59/2, 's_5': 29, 's_6': 57/2, 's_7': 28, 's_8': 55/2, 's_9': 63/2}
J = {('s_1', 's_2'): 4.5, ('s_1', 's_3'): 4.5, ('s_1', 's_4'): 4.5, ('s_1', 's_5'): 4.5, ('s_1', 's_6'): 4.5, ('s_1', 's_7'): 4.5,
    ('s_1', 's_8'): 4.5, ('s_1', 's_9'): 4.5, ('s_2', 's_3'): 4.5, ('s_2', 's_4'): 4.5, ('s_2', 's_5'): 4.5, ('s_2', 's_6'): 4.5, 
    ('s_2', 's_7'): 4.5, ('s_2', 's_8'): 4.5, ('s_2', 's_9'): 4.5, ('s_3', 's_4'): 4.5, ('s_3', 's_5'): 4.5, ('s_3', 's_6'): 4.5, 
    ('s_3', 's_7'): 4.5, ('s_3', 's_8'): 4.5, ('s_1', 's_9'): 4.5, ('s_4', 's_5'): 4.5, ('s_4', 's_6'): 4.5, ('s_4', 's_7'): 4.5, 
    ('s_4', 's_8'): 4.5, ('s_4', 's_9'): 4.5, ('s_5', 's_6'): 4.5, ('s_5', 's_7'): 4.5, ('s_5', 's_8'): 4.5, ('s_5', 's_9'): 4.5, 
    ('s_6', 's_7'): 4.5, ('s_6', 's_8'): 4.5, ('s_6', 's_9'): 4.5, ('s_7', 's_8'): 4.5, ('s_7', 's_9'): 4.5, ('s_8', 's_9'): 4.5,}

sampleset = sampler.sample_ising(h, J, num_reads=5000, label='knapsack problem - 8 objects')

print(sampleset)
print(sampleset.info)

dwave.inspector.show(sampleset)