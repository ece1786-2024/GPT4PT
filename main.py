# from ie_agent import IeClassifier
# from sn_agent import SnClassifier
from tf_agent.agent import TfClassifier
# from jp_agent import JpClassifier


# Introvert (I) and Extrovert (E)
# c1 = IeClassifier().start()

# Sensing (S) and Intuition (N)
# c2 = SnClassifier().start()

# Thinking (T) and Feeling (F)

c3 = TfClassifier().start()

# Judging (J) and Perceiving (P)
#c4 = JpClassifier().start()

#print(f'Final prediction: {c1}{c2}{c3}{c4}')
