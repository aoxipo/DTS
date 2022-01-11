from houglass.models.posenet import MYDensityNet as Net
from data_generate import  DataGenerate
from torch.autograd import Variable
from torchsummary import summary
import os
import torch
import numpy as np
import datetime
from torch.utils.data import DataLoader,TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_gpu = torch.cuda.is_available()
print("use gpu:", use_gpu)
class Train():
    def __init__(self, in_channles, out_channels, is_show = True):
        self.in_channels = in_channles
        self.out_channels = out_channels
        self.lr = 0.0001
        self.history_acc = []
        self.history_loss = []
        self.history_test_acc = []
        
        self.create(is_show)
    
    def create(self, is_show):
        stack = 1
        self.model = Net( stack, self.in_channels, self.out_channels)
        
        if(is_show):
            self.model.summary()
            
        self.cost = torch.nn.MSELoss()
        if(use_gpu):
            self.model = self.model.cuda()
            self.cost = self.cost.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        
    def train(self, n_epochs, data_loader_train, data_loader_test, batch_size = 32):
        for epoch in range(n_epochs):
            running_loss = 0.0
            running_correct = 0
            print("Epoch {}/{}".format(epoch, n_epochs))
            print("-"*10)
            train_index = 0
            test_index = 0
     
            for data in data_loader_train:
                X_train, y_train = data
                X_train, y_train = Variable(X_train), Variable(y_train)
                if(use_gpu):
                    X_train = X_train.to(device)
                    y_train = y_train.to(device)
                    
                self.optimizer.zero_grad()
             
                outputs = self.model(X_train)

                loss = self.cost(outputs, y_train)
                loss.backward()
                self.optimizer.step()
                # print(loss.data.item())
                if(use_gpu):
                    running_loss += loss.cpu().data.item()
                    running_correct +=  loss.cpu().data.item()
                else:
                    running_loss += loss.data.item()
                    running_correct +=  loss.data.item()
                # print("Acc: ",torch.sum(pred == y_train.data).item())
                train_index += 1
                
            
            testing_correct = 0
        
            for data in data_loader_test:
                X_test, y_test = data
                
                if(use_gpu):
                    X_test = X_test.to(device)
                    y_test = y_test.to(device)
                
               
                outputs = self.model(X_test)
                #_, pred = torch.max(outputs.data, 1)
                loss = self.cost(outputs, y_test)
                if(use_gpu):
                    testing_correct += loss.cpu().data.item()
                else:
                    testing_correct += loss.data.item()
                test_index += 1
        
            epoch_loss = running_loss/train_index 
            epoch_acc = running_correct/train_index * 100
            epoch_test_acc = testing_correct/test_index * 100
            
            self.history_acc.append(epoch_acc)
            self.history_loss.append(epoch_loss)
            self.history_test_acc.append(epoch_test_acc)
            print(
                "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(
                    epoch_loss,
                    epoch_acc,
                    epoch_test_acc
                )
             )
        self.save_parameter()
        self.save_history()
        
    def predict(self, image):
        pass
        self.model.eval()
        if(len(image.shape) != 4):
            image = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
        output = self.model( Variable(image))
        _, preds = torch.max(output, 1)
        print(preds)
        return preds
    
    def save_history(self, file_path = './save/'):
        if not os.path.exists(file_path): 
            os.mkdir(file_path)
        fo = open(file_path + "loss_history.txt", "w+")
        fo.write(str(self.history_loss))
        fo.close()
        fo = open(file_path + "acc_history.txt", "w+")
        fo.write(str(self.history_acc))
        fo.close()
        fo = open(file_path + "test_history.txt", "w+")
        fo.write(str(self.history_test_acc))
        fo.close()   
        
    def save_parameter(self, file_path = './save/'):
        if not os.path.exists(file_path): 
            os.mkdir(file_path)
        file_path = file_path + "model_" +str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace("-","_").replace(".","_") + ".pkl"
        torch.save(obj=self.model.state_dict(), f=file_path)
    def load_parameter(self, file_path = './save/' ):
        # self.model.load_state_dict(torch.load('model_parameter.pkl'))
        self.model.load_state_dict(torch.load(file_path))
    

MYDataLoader = DataGenerate(data_set_number = "L")
data_dict = MYDataLoader.get_train_dataset()

train_data_x = data_dict["train_data_x"]
train_data_anchor = data_dict["train_data_y"]["cord"]
train_data_density = data_dict["train_data_y"]["density"]

test_data_x = data_dict["test_data_x"]
test_data_anchor = data_dict["test_data_y"]["cord"]
test_data_density = data_dict["test_data_y"]["density"]

val_data_x = data_dict["val_data_x"]
val_data_anchor  = data_dict["val_data_y"]["cord"]
val_data_density = data_dict["val_data_y"]["density"]


traindata = TensorDataset( torch.from_numpy(np.expand_dims(train_data_x,1)),torch.from_numpy(np.expand_dims(train_data_density, 1)))
train_dataloader = DataLoader(
    dataset = traindata,
    batch_size = 32,
    shuffle = True
    #num_workers = 3,
)

valdata = TensorDataset( torch.from_numpy(np.expand_dims(val_data_x,1)),torch.from_numpy(np.expand_dims(val_data_density, 1)))
val_dataloader = DataLoader(
    dataset = valdata,
    batch_size = 32,
    shuffle = True
    #num_workers = 3,
)


trainer = Train(255, 1, False)
trainer.train(500, train_dataloader, val_dataloader)