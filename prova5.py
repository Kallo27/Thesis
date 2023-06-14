from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s_1': 0.5, 's_2': 0.0, 's_3': 1.0, 's_4': 2.0, 's_5': 1.0}
J = {('s_1', 's_2'): 1.5, ('s_2', 's_3'): 1.5, ('s_1', 's_3'): 1.5, ('s_1', 's_4'): 1.5, ('s_1', 's_5'): 1.5, 
    ('s_2', 's_4'): 1.5, ('s_2', 's_5'): 1.5,('s_3', 's_4'): 1.5, ('s_3', 's_5'): 1.5, ('s_4', 's_5'): 1.5}
sampleset = sampler.sample_ising(h, J, num_reads=2000, label='knapsack problem - 2 objects')
print(sampleset)
print(sampleset.info)
dwave.inspector.show(sampleset)