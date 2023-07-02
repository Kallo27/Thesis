from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s_1': 1, 's_2': 1/2, 's_3': 3/2}
J = {('s_1', 's_2'): 1.5, ('s_1', 's_3'): 1.5, ('s_2', 's_3'): 1.5}

sampleset = sampler.sample_ising(h, J, num_reads=5000, label='knapsack problem - 2 objects')

print(sampleset)
print(sampleset.info)

dwave.inspector.show(sampleset)