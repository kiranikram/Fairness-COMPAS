"""## Training of Different models

### Training of NN_Baseline
"""

time0 = time()
epochs = 100
test_accuracies_Baseline = []
train_accuracies_Baseline = []
conf_matrixes_Baseline = []

for e in range(epochs):
    running_loss_Baseline = 0
    for features, labels in trainloader:
        
        # Training pass
        optimizer_Baseline.zero_grad()
        
        output = NN_Baseline(features)
        exp_output = torch.exp(output)

        lbls = labels.view(labels.shape[0])
        loss = criterion_Baseline(exp_output,lbls)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer_Baseline.step()
        
        running_loss_Baseline += loss.mean().item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss_Baseline/len(trainloader)))
        conf_matrix_test = get_accuracies(NN_Baseline, priors = False, test_data = True)
        test_accuracy = np.sum(np.diag(conf_matrix_test))/np.sum(conf_matrix_test)
        conf_matrix_train = get_accuracies(NN_Baseline, priors = False, test_data = False)
        train_accuracy = np.sum(np.diag(conf_matrix_train))/np.sum(conf_matrix_train)

        test_accuracies_Baseline.append(test_accuracy)
        train_accuracies_Baseline.append(train_accuracy)
        conf_matrixes_Baseline.append(conf_matrix_test)


print("\nTraining Time (in minutes) =",(time()-time0)/60)

"""### Training of NN_WDecay"""

time0 = time()
epochs = 100
test_accuracies_WDecay = []
train_accuracies_WDecay = []
conf_matrixes_Wdecay = []

for e in range(epochs):
    running_loss_WDecay = 0
    for features, labels in trainloader:
        
        # Training pass
        optimizer_WDecay.zero_grad()
        
        output = NN_WDecay(features)
        exp_output = torch.exp(output)

        lbls = labels.view(labels.shape[0])
        loss = criterion_WDecay(exp_output,lbls)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer_WDecay.step()
        
        running_loss_WDecay += loss.mean().item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss_WDecay/len(trainloader)))
        conf_matrix_test = get_accuracies(NN_WDecay, priors = False, test_data = True)
        test_accuracy = np.sum(np.diag(conf_matrix_test))/np.sum(conf_matrix_test)
        conf_matrix_train = get_accuracies(NN_WDecay, priors = False, test_data = False)
        train_accuracy = np.sum(np.diag(conf_matrix_train))/np.sum(conf_matrix_train)

        test_accuracies_WDecay.append(test_accuracy)
        train_accuracies_WDecay.append(train_accuracy)
        conf_matrixes_Wdecay.append(conf_matrix_test)

print("\nTraining Time (in minutes) =",(time()-time0)/60)

"""### Training of NN_WDecay_Loss"""

time0 = time()
epochs = 100
test_accuracies_WDecay_Loss = []
train_accuracies_WDecay_Loss = []
conf_matrixes_Wdecay_Loss = []

for e in range(epochs):
    running_loss_WDecay_Loss = 0
    for features, labels in trainloader:
        
        # Training pass
        optimizer_WDecay_Loss.zero_grad()
        
        output = NN_WDecay_Loss(features)
        exp_output = torch.exp(output)

        lbls = labels.view(labels.shape[0])
        loss = criterion_WDecay_Loss(exp_output,lbls)
        
        #This is where the model learns by backpropagating
        loss.mean().backward()
        
        #And optimizes its weights here
        optimizer_WDecay_Loss.step()
        
        running_loss_WDecay_Loss += loss.mean().item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss_WDecay_Loss/len(trainloader)))
        conf_matrix_test = get_accuracies(NN_WDecay_Loss, priors = False, test_data = True)
        test_accuracy = np.sum(np.diag(conf_matrix_test))/np.sum(conf_matrix_test)
        conf_matrix_train = get_accuracies(NN_WDecay_Loss, priors = False, test_data = False)
        train_accuracy = np.sum(np.diag(conf_matrix_train))/np.sum(conf_matrix_train)

        test_accuracies_WDecay_Loss.append(test_accuracy)
        train_accuracies_WDecay_Loss.append(train_accuracy)
        conf_matrixes_Wdecay_Loss.append(conf_matrix_test)

print("\nTraining Time (in minutes) =",(time()-time0)/60)

"""### Training of NN_WDecay_Loss_Priors"""

time0 = time()
epochs = 100
test_accuracies_WDecay_Loss_Priors = []
train_accuracies_WDecay_Loss_Priors = []
conf_matrixes_Wdecay_Loss_Priors = []

for e in range(epochs):
    running_loss_WDecay_Loss_Priors = 0
    for features, labels in trainloader_Priors:
        
        # Training pass
        optimizer_WDecay_Loss_Priors.zero_grad()
        
        output = NN_WDecay_Loss_Priors(features)
        exp_output = torch.exp(output)

        lbls = labels.view(labels.shape[0])
        loss = criterion_WDecay_Loss_Priors(exp_output,lbls)
        
        #This is where the model learns by backpropagating
        loss.mean().backward()
        
        #And optimizes its weights here
        optimizer_WDecay_Loss_Priors.step()
        
        running_loss_WDecay_Loss_Priors += loss.mean().item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss_WDecay_Loss_Priors/len(trainloader)))
        conf_matrix_test = get_accuracies(NN_WDecay_Loss_Priors, priors = False, test_data = True)
        test_accuracy = np.sum(np.diag(conf_matrix_test))/np.sum(conf_matrix_test)
        conf_matrix_train = get_accuracies(NN_WDecay_Loss_Priors, priors = False, test_data = False)
        train_accuracy = np.sum(np.diag(conf_matrix_train))/np.sum(conf_matrix_train)

        test_accuracies_WDecay_Loss_Priors.append(test_accuracy)
        train_accuracies_WDecay_Loss_Priors.append(train_accuracy)
        conf_matrixes_Wdecay_Loss_Priors.append(conf_matrix_test)

print("\nTraining Time (in minutes) =",(time()-time0)/60)

"""## Accuracy and Confusion Matrix of model with feature engineering of priors, loss weights, and weight decay

## Confusion Matrix and Precision/Recall of N_Baseline
"""

import seaborn as sns
cm = get_accuracies(NN_Baseline, priors = False, test_data = True)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%')

cm = get_accuracies(NN_Baseline, priors = False, test_data = True)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
##
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

accuracy_Baseline = np.sum(np.diag(cm))/np.sum(cm)
print('pure accuracy', accuracy_Baseline)
recall_Baseline = np.diag(cm) / np.sum(cm, axis = 1)
print('recall per class', recall_Baseline)
precision_Baseline = np.diag(cm) / np.sum(cm, axis = 0)
print('precision per class', precision_Baseline)

"""## Confusion Matrix and Precision/Recall of NN_WDecay"""

import seaborn as sns
cm = get_accuracies(NN_WDecay, priors = False, test_data = True)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%')

cm = get_accuracies(NN_WDecay, priors = False, test_data = True)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
##
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

accuracy_WDecay = np.sum(np.diag(cm))/np.sum(cm)
print('pure accuracy', accuracy_WDecay)
recall_WDecay = np.diag(cm) / np.sum(cm, axis = 1)
print('recall per class', recall_WDecay)
precision_WDecay = np.diag(cm) / np.sum(cm, axis = 0)
print('precision per class', precision_WDecay)

"""## Confusion Matrix and Precision/Recall of NN_WDecay_Loss"""

import seaborn as sns
cm = get_accuracies(NN_WDecay_Loss, priors = False, test_data = True)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%')

cm = get_accuracies(NN_WDecay_Loss, priors = False, test_data = True)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
##
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

accuracy_WDecay_Loss = np.sum(np.diag(cm))/np.sum(cm)
print('pure accuracy', accuracy_WDecay_Loss)
recall_WDecay_Loss = np.diag(cm) / np.sum(cm, axis = 1)
print('recall per class', recall_WDecay_Loss)
precision_WDecay_Loss = np.diag(cm) / np.sum(cm, axis = 0)
print('precision per class', precision_WDecay_Loss)

"""## Confusion Matrix and Precision/Recall of NN_WDecay_Loss_Priors"""

import seaborn as sns
cm = get_accuracies(NN_WDecay_Loss_Priors, priors = True, test_data = True)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%')

cm = get_accuracies(NN_WDecay_Loss_Priors, priors = True, test_data = True)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
##
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

accuracy_WDecay_Loss_Priors = np.sum(np.diag(cm))/np.sum(cm)
print('pure accuracy', accuracy_WDecay_Loss_Priors)
recall_WDecay_Loss_Priors = np.diag(cm) / np.sum(cm, axis = 1)
print('recall per class', recall_WDecay_Loss_Priors)
precision_WDecay_Loss_Priors = np.diag(cm) / np.sum(cm, axis = 0)
print('precision per class', precision_WDecay_Loss_Priors)