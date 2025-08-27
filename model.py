import torch
import torch.nn as nn
import numpy as np
import pickle

# Reusable Class wiht layer structure has been defined here
class HiddenLayer(nn.Module):
    def __init__(self,NumNeurons,NumOutput,DropoutRate):
        super().__init__()
        #first layer and input layer
        self.Layer=nn.Linear(NumNeurons,out_features=NumOutput)
        #batch normalization layer
        self.batchnorm=nn.BatchNorm1d(NumOutput)
        #relu activation function layer
        self.activation=nn.ReLU()
        #dropout layer
        self.dropout=nn.Dropout1d(p=DropoutRate)
    #defining forward method for reusable class
      #forward method is defined
    def forward(self,x):
        x=self.Layer(x)
        x=self.batchnorm(x)
        x=self.activation(x)
        x=self.dropout(x)
        return x
# Building my main model class
class HousePricePrediction(nn.Module):
    #calling initiator with required parameters
    def __init__(self,num_inputs,num_out_features):
        #parent class initiator method is being called
        super().__init__()
        # Define InputLayer
        self.inputlayer=nn.Linear(in_features=num_inputs,out_features=200)
        # Define Hidden Layers
        self.hidden1=HiddenLayer(200,50,0.1)
        self.hidden2=HiddenLayer(50,150,0.3)
        self.hidden3=HiddenLayer(150,100,0.2)
        self.hidden4=HiddenLayer(100,50,0.2)
        self.hidden5=HiddenLayer(50,30,0.1)
        self.hidden6=HiddenLayer(30,20,0.0)
        #last final layer, the output layer
        self.outputlayer=nn.Linear(20,out_features=num_out_features)
    #forward method is defined
    def forward(self,x):
        x=self.inputlayer(x)
        x=self.hidden1(x)
        x=self.hidden2(x)
        x=self.hidden3(x)
        x=self.hidden4(x)
        x=self.hidden5(x)
        x=self.hidden6(x)
        x=self.outputlayer(x)
        return x
# function for prediction
def predict(input):
    # Creating model object
    # Importing preprocessors
    with open(f'preprocessorx.pkl', 'rb') as f:
        preprocessor_X = pickle.load(f)
    with open(f'preprocessory.pkl', 'rb') as f:
        preprocessor_Y = pickle.load(f)
    input_size = 17
    output_size = 1
    HousePricePrediction_model = HousePricePrediction(num_inputs=input_size, num_out_features=output_size).to(torch.device('cpu'))
    states = torch.load("HousePricePrediction.pth", map_location=torch.device('cpu'))
    HousePricePrediction_model.load_state_dict(states)
    HousePricePrediction_model.eval()
    processed_input=preprocessor_X.transform(input)
    print(processed_input)
    processed_input=torch.from_numpy(processed_input).float()
    with torch.no_grad():
        output=HousePricePrediction_model(processed_input)
        output=output.detach().cpu().numpy()
        output=preprocessor_Y.inverse_transform(output)
        output=np.expm1(output)
        return output


