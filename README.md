# Fairness-COMPAS
Attempt to ensure fairness in ML Algorithms that decide on Recidivism risk. 

To explore algorithmic fairness particular attention was paid to the criminal justice system, where statistical tools are used to make decisions regarding defendants about sentencing, paroles and bail (Klienberg et al., 2016). In a report that contributed to heightened awareness about potential vulner- abilities of statistical models for prediction, Angwin et al.(2016), displayed that a risk tool developed by the American company Northepointe Inc (Northpointe Inc., 2015), the COMPAS algorithm , showed proof of racial discrimination. Particularly, they looked at data collected and made available ProPublica that the COMPAS algorithm misclassified the risk of recidivism depending on whether the defendant was African American or Caucasian.

A recent foray into the performance and nature of these algorithmic techniques however, has revealed the vulnerability of apparent accuracy for a given prediction. Values for accuracy, from a model perspective it seems, does not account for levels of fairness or inherent bias such models may subsume.The aim wass to produce a system, using machine learning techniques, which will accurately predict a defendant’s risk of recidivism, while mitigating any bias.

Implemented:
1. To address the issue of misclassification , Khan et al, (2017) propose the implementation of a ’cost- matrix’ applied to the vector of class predictions. The cost matrix, using Bayesian principles of poste- rior probability, is intended to modify the output of the last layer, ascribing a cost to misclassification proportional to distribution of of the target features in the dataset. Necessary to the computation of this cost function is the calculation of posterior probabilities of the occurrence of a particular label, which in this case is the potential of Recividism, Violent Recividism and No Recividism. Using this method, the vector of predicted labels VL would be updated via:

ActualCS−Y = ξpqP(p|q) ∗ ActualY Where,ξpqP(p|q)

Where,ξpqP(p|q)is the ratio of each class occurring to the total number of examples and ActualCS−Y
is the amended vector of labels, rescaled by the matrix of prior probabilities.

A simpler way to incorporate a cost is to include in the loss function a 1-D tensor of weights that account for the ratios of samples to class instances. This method is more effective than using an entire matrix of posterior probabilities as it simply over-penalizes the misclassification from within the NLLLoss function.

2. In a ruling on State vs Loomis presented to the Wisconsin Supreme Court, the Court ruled that the usage of such machine learning algorithms had to occur in conjunction with judgements conferred by those in the authority to decide on incarceration (Donohue , 2019). One of the points highlighted as an outcome of this examination was that greater likelihood must be ascribed to those individuals with a history of criminal activity. Due to its ordinal encoding, the model was not taking into account this heightened likelikood. Hence ascribed to it were the features relating Juvenile Felony Counts, Juvenile Misdemeanor Counts, Juvenile ’Other’ Counts and Priors Counts an emphasis factor equivalent to the mean of the distribution of each feature to the instances where there were no priors.


Created with Sofoklis K for City, University of London CW 
