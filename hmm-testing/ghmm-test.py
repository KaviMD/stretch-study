# %%
import ghmm

# %%
sigma = ghmm.IntegerRange(1,7)

train_seq = ghmm.SequenceSet(sigma, [[1,1,1,1,3,3,1,1,1,1,3,3,3,3,3,3,3,1,1,1,1,1]])

A = [[0.99,0.01],[0.99, 0.01]]

B = [[1.0/6]*6]*2

pi = [0.5] * 2

m = ghmm.HMMFromMatrices(sigma, ghmm.DiscreteDistribution(sigma), A, B, pi)

m.baumWelch(train_seq, 100000000, 0.000000000000001)

print(m.asMatrices())
# %%
print(map(sigma.external, m.sampleSingle(20)))
# %%
v = m.viterbi(test_seq)
print v

# %%
my_seq = ghmm.EmissionSequence(sigma, [1] * 20 + [6] * 10 + [1] * 40)
print m.viterbi(my_seq)