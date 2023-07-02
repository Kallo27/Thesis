from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s_1': 9/2, 's_2': 4.0, 's_3': 7/2, 's_4': 5.0}
J = {('s_1', 's_2'): 2.5, ('s_1', 's_3'): 2.5, ('s_1', 's_4'): 2.5, ('s_2', 's_3'): 2.5, 
    ('s_2', 's_4'): 2.5, ('s_3', 's_4'): 2.5}

sampleset = sampler.sample_ising(h, J, num_reads=5000, label='knapsack problem - 3 objects')

print(sampleset)
print(sampleset.info)

dwave.inspector.show(sampleset)