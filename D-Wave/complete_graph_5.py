from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s_1': 7.0, 's_2': 13/2, 's_3': 6.0, 's_4': 11/2, 's_5': 15/2}
J = {('s_1', 's_2'): 2.5, ('s_2', 's_3'): 2.5, ('s_1', 's_3'): 2.5, ('s_1', 's_4'): 2.5, ('s_1', 's_5'): 2.5, 
    ('s_2', 's_4'): 2.5, ('s_2', 's_5'): 2.5,('s_3', 's_4'): 2.5, ('s_3', 's_5'): 2.5, ('s_4', 's_5'): 2.5}

sampleset = sampler.sample_ising(h, J, num_reads=5000, label='knapsack problem - 4 objects', chain_strength = 2)

print(sampleset)
print(sampleset.info)

dwave.inspector.show(sampleset)