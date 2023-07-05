from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s_1': 313/4, 's_2': 971/4, 's_3': 1669/4, 's_4': 2407/4, 's_5': 3185/4, 's_6': 315/2, 's_7': 160, 's_8': 330, 's_9': 700}
J = {('s_1', 's_2'): 15/2, ('s_1', 's_3'): 12.5, ('s_1', 's_4'): 35.5, ('s_1', 's_5'): 22.5, ('s_1', 's_6'): 2.5, ('s_1', 's_7'): 5,
    ('s_1', 's_8'): 10, ('s_1', 's_9'): 20, ('s_2', 's_3'): 37.5, ('s_2', 's_4'): 52.5, ('s_2', 's_5'): 135/2, ('s_2', 's_6'): 7.5, 
    ('s_2', 's_7'): 15, ('s_2', 's_8'): 30, ('s_2', 's_9'): 60, ('s_3', 's_4'): 175/2, ('s_3', 's_5'): 225/2, ('s_3', 's_6'): 25/2, 
    ('s_3', 's_7'): 25, ('s_3', 's_8'): 50, ('s_1', 's_9'): 100, ('s_4', 's_5'): 315/2, ('s_4', 's_6'): 35/2, ('s_4', 's_7'): 35, 
    ('s_4', 's_8'): 70, ('s_4', 's_9'): 140, ('s_5', 's_6'): 45/2, ('s_5', 's_7'): 45, ('s_5', 's_8'): 90, ('s_5', 's_9'): 180, 
    ('s_6', 's_7'): 5, ('s_6', 's_8'): 10, ('s_6', 's_9'): 20, ('s_7', 's_8'): 20, ('s_7', 's_9'): 40, ('s_8', 's_9'): 80}

sampleset = sampler.sample_ising(h, J, num_reads=5000, label='knapsack problem - 8 objects')

print(sampleset)
print(sampleset.info)

dwave.inspector.show(sampleset)