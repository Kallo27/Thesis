from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dwave.inspector
sampler = AutoEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))

h = {'s1': 0.5, 's2': 0.0, 's3': 1.0, 's4': 2.0, 's5': 1.0, 's6': 0.5, 's7': 1.0, 's8': 1.0, 's9': 2.0, 's0': 1.5}

J = {('s1', 's2'): 1.5, ('s1', 's3'): 1.5, ('s1', 's4'): 1.5, ('s1', 's5'): 1.5, ('s1', 's6'): 1.5, 
    ('s1', 's7'): 1.5, ('s1', 's8'): 1.5, ('s1', 's9'): 1.5, ('s1', 's0'): 1.5, ('s2', 's3'): 1.5, 
    ('s2', 's4'): 1.5, ('s2', 's5'): 1.5, ('s2', 's6'): 1.5, ('s2', 's7'): 1.5, ('s2', 's8'): 1.5, 
    ('s2', 's9'): 1.5, ('s2', 's0'): 1.5, ('s3', 's4'): 1.5, ('s3', 's5'): 1.5, ('s3', 's6'): 1.5, 
    ('s3', 's7'): 1.5, ('s3', 's8'): 1.5, ('s3', 's9'): 1.5, ('s3', 's0'): 1.5, ('s4', 's5'): 1.5, 
    ('s4', 's6'): 1.5, ('s4', 's7'): 1.5, ('s4', 's8'): 1.5, ('s4', 's9'): 1.5, ('s4', 's0'): 1.5, 
    ('s5', 's6'): 1.5, ('s5', 's7'): 1.5, ('s5', 's8'): 1.5, ('s5', 's9'): 1.5, ('s5', 's0'): 1.5,
    ('s6', 's7'): 1.5, ('s6', 's8'): 1.5, ('s6', 's9'): 1.5, ('s6', 's0'): 1.5, ('s7', 's8'): 1.5, 
    ('s7', 's9'): 1.5, ('s7', 's0'): 1.5, ('s8', 's9'): 1.5, ('s8', 's0'): 1.5, ('s9', 's0'): 1.5}


sampleset = sampler.sample_ising(h, J, num_reads=2000, label='complete graph - 10 qubits')

print(sampleset)
print(sampleset.info)
dwave.inspector.show(sampleset)