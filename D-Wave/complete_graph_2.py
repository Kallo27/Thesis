from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s_1': -5, 's_2': -5}
J = {('s_1', 's_2'): 2.5}

sampleset = sampler.sample_ising(h, J, num_reads=5000, label='knapsack problem - 1 object')

print(sampleset)
print(sampleset.info)

dwave.inspector.show(sampleset)