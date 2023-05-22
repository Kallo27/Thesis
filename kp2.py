from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector
sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))

h = {'s_1': 0.5, 's_2': 0.0, 's_3': 1.0}
J = {('s_1', 's_2'): 1.5, ('s_2', 's_3'): 1.5, ('s_2', 's_3'): 1.5}
sampleset = sampler.sample_ising(h, J, num_reads=2000, label='knapsack problem - 2 objects')
print(sampleset)
dwave.inspector.show(sampleset)