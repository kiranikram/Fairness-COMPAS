#setting up our dataloaders and splitting our dataset into test and train
trainloader = torch.utils.data.DataLoader(Pixie_NN_train, batch_size = 32)
testloader = torch.utils.data.DataLoader(Pixie_NN_test, batch_size=32)

trainloader_Priors = torch.utils.data.DataLoader(Pixie_NN_train_priors, batch_size = 32)
testloader_Priors = torch.utils.data.DataLoader(Pixie_NN_test_priors, batch_size=32)

#Architecture of the NN models: three hidden layers of 20,15,15 nodes each
input_size = 9
hidden_layer_sizes=[20,15,15,10]
output_size = 3

#Baseline Neural Network without feature engineering of priors, weight decays, or weightage in the loss function
NN_Baseline = nn.Sequential(nn.Linear(input_size, hidden_layer_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[2], hidden_layer_sizes[3]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[3], output_size),
                      nn.LogSoftmax(dim=1))
optimizer_Baseline = torch.optim.Adam(NN_Baseline.parameters())
criterion_Baseline = nn.NLLLoss()

#Neural Network with weight decay, but no feature engineering of priors or weightage in the loss function
NN_WDecay = nn.Sequential(nn.Linear(input_size, hidden_layer_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[2], hidden_layer_sizes[3]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[3], output_size),
                      nn.LogSoftmax(dim=1))
optimizer_WDecay = torch.optim.Adam(NN_WDecay.parameters(), weight_decay = 0.001)
criterion_WDecay = nn.NLLLoss()

#Neural Network with weight decay and weightage in the loss function, but no feature engineering
NN_WDecay_Loss = nn.Sequential(nn.Linear(input_size, hidden_layer_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[2], hidden_layer_sizes[3]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[3], output_size),
                      nn.LogSoftmax(dim=1))
optimizer_WDecay_Loss = torch.optim.Adam(NN_WDecay_Loss.parameters(), weight_decay = 0.001)
criterion_WDecay_Loss = nn.NLLLoss(weight = torch.tensor([1.94,2.71,8.62]), reduction='none') #[1.94,2.71,8.62] which are ratios of total samples over num of instances

#Neural Network with weight decay, weightage in the loss function and feature engineering
NN_WDecay_Loss_Priors = nn.Sequential(nn.Linear(input_size, hidden_layer_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[2], hidden_layer_sizes[3]),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_sizes[3], output_size),
                      nn.LogSoftmax(dim=1))
optimizer_WDecay_Loss_Priors = torch.optim.Adam(NN_WDecay_Loss_Priors.parameters(), weight_decay = 0.001)
criterion_WDecay_Loss_Priors = nn.NLLLoss(weight = torch.tensor([1.94,2.71,8.62]), reduction='none') #[1.94,2.71,8.62] which are ratios of total samples over num of instances

"""# Accuracy Functions"""

#Helper function which computes test and train accuracies
def get_accuracies(model, priors = False, test_data = True):
    nn_model = model
    if test_data:
        if priors: dataldr = testloader_Priors
        else: dataldr = testloader
    else:
        if priors: dataldr = trainloader_Priors
        else: dataldr = trainloader

    conf_matrix = np.zeros((3,3))
    zeros_total, zeros_correct, ones_total, ones_correct, twos_total,twos_correct = 0,0,0,0,0,0
    for features,labels in dataldr:
        with torch.no_grad():
            logps = nn_model(features)

        ps = torch.exp(logps)
        pred_label = torch.squeeze(torch.argmax(ps,dim=1))
        true_label = torch.squeeze(labels.T)
        conf_matrix += confusion_matrix(true_label, pred_label, labels=[0,1,2])
    
    return conf_matrix