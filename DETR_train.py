from detr_.models.detr import *
from data_generate import DataGenerate
from torch.autograd import Variable
from torchsummary import summary
import os
import torch
import numpy as np
import datetime
from torch.utils.data import DataLoader,TensorDataset
import cv2
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_gpu = torch.cuda.is_available()
print("use gpu:", use_gpu)


class Train():
    def __init__(self, 
            hidden_dim = 256,
            dropout = 0.1,
            nheads = 8,
            dim_feedforward = 2048,
            enc_layers = 6,
            dec_layers = 6,
            pre_norm = False,
            is_show = False,
            class_number = 1,
            require_number = 24,
            image_shape = (128,128),
            ):

        self.lr = 0.00001
        self.image_shape = image_shape
        self.history_acc = []
        self.history_loss = []
        self.history_test_acc = []
        self.history_test_loss = []
        self.require_number = require_number
        self.class_number = class_number
        self.trainsformer = build_easy_transformer(
            hidden_dim ,
            dropout ,
            nheads ,
            dim_feedforward ,
            enc_layers ,
            dec_layers ,
            pre_norm ,
        )
        self.create(is_show)
    
    def create(self, is_show):
        self.model = MYDETR( self.trainsformer, self.class_number, self.require_number)
        if(is_show):
            self.model.summary()
        self.cost_class = torch.nn.CrossEntropyLoss()
        #self.cost_cord = torch.nn.SmoothL1Loss()
        #self.cost_cord = torch.nn.MSELoss(reduce=True, size_average=True)
        self.cost_cord = torch.nn.L1Loss()
        if(use_gpu):
            self.model = self.model.cuda()
            self.cost_class = self.cost_class.cuda()
            self.cost_cord = self.cost_cord.cuda()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = 0.01)
        
    def train(self, n_epochs, data_loader_train, data_loader_test):
        for epoch in range(n_epochs):
            start_time = datetime.datetime.now()
            running_loss = 0.0
            running_correct = 0
            print("Epoch {}/{}".format(epoch, n_epochs))
            print("-"*10)
            train_index = 1
            test_index = 1
            class_loss_train = 0
            crod_loss_train = 0
            for data in data_loader_train:
                X_train, X_pos, X_mask, x_anchor, x_class  = data
                X_train, X_pos, X_mask, x_anchor, x_class = Variable(X_train), Variable(X_pos), Variable(X_mask), Variable(x_anchor), Variable(x_class)
                if(use_gpu):
                    X_train = X_train.to(device)
                    X_pos = X_pos.to(device)
                    X_mask = X_mask.to(device)
                    x_anchor = x_anchor.to(device)
                    x_class = x_class.to(device)
                    
                
                
                outputs = self.model(X_train, X_pos, X_mask)
                
                _,pred = torch.max(outputs['pred_logits'], 2)
                _,pred_label = torch.max(x_class, 2)
                #  y_train = y_train.long()

                x_class = x_class.float()
                total_loss_class = 0
                total_loss_cord = 0
                for j in range(batch_size):
                    for i in range(self.require_number):
                        total_loss_class += self.cost_class( outputs["pred_logits"][j][i].unsqueeze(0), x_class[j][i].unsqueeze(0))
                        for i_cord in range(3):
                            total_loss_cord += self.cost_cord( outputs["pred_boxes"][j][i][i_cord], x_anchor[j][i][i_cord])
                #loss1 = self.cost_class(outputs['pred_logits'], x_class)
                #loss2 = self.cost_cord(outputs['pred_boxes'], x_anchor) 
                
                # loss1.backward()
                # loss2.backward()
                loss = total_loss_class + total_loss_cord
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc = [ pred_label[i] == pred[i] for i in range(len(pred)) ]
                # print(torch.sum(acc[0]).cpu().item())
                # print(loss.data.item())
                
                if(use_gpu):
                    running_loss += loss.cpu().data.item()
                    class_loss_train += total_loss_class.cpu().data.item()
                    crod_loss_train += total_loss_cord.cpu().data.item()
                    running_correct += torch.sum(acc[0]).cpu().item()/(len(pred)*self.require_number)
                else:
                    running_loss += loss.data.item()
                    class_loss_train += total_loss_class.data.item()
                    crod_loss_train += total_loss_cord.data.item()
                    running_correct += torch.sum(acc[0]).item()/(len(pred)*self.require_number)
                # print( torch.sum(acc[0]).item()/(len(pred)*24))
                # print("Acc: ",torch.sum(pred == y_train.data).item())
                train_index += 1
            #if epoch > 75: 
            #    self.save_parameter("./save/save_epoch_"+str(epoch)+"/")
            
            testing_loss = 0
            testing_correct = 0
            class_loss_test = 0
            crod_loss_test = 0
            # for data in data_loader_test:
            #     X_test, X_pos, X_mask, y_anchor, y_class = data
            #     X_test, X_pos, X_mask, y_anchor, y_class = Variable(X_test), Variable(X_pos), Variable(X_mask), Variable(y_anchor), Variable(y_class)
            #     if(use_gpu):
            #         X_test = X_train.to(device)
            #         X_pos = X_pos.to(device)
            #         X_mask = X_mask.to(device)
            #         y_anchor = y_anchor.to(device)
            #         y_class = y_class.to(device)
                
            #     outputs = self.model(X_test, X_pos, X_mask)
                
            #     _,pred = torch.max(outputs['pred_logits'], 2)
            #     _,pred_label = torch.max(y_class, 2)
                
            #     y_class = y_class.float()
                
            #     total_loss_class_test = 0
            #     total_loss_cord_test = 0
            #     for j in range(batch_size):
            #         for i in range(self.require_number):
            #             total_loss_class_test += self.cost_class( outputs["pred_logits"][j][i].unsqueeze(0), y_class[j][i].unsqueeze(0))
            #             for i_cord in range(3):
            #                 total_loss_cord_test += self.cost_cord( outputs["pred_boxes"][j][i][i_cord], y_anchor[j][i][i_cord])
            #     #loss1 = self.cost_class(outputs['pred_logits'], x_class)
            #     #loss2 = self.cost_cord(outputs['pred_boxes'], x_anchor) 
            #     loss = total_loss_class_test + total_loss_cord_test
            #     acc = [ pred_label[i] == pred[i] for i in range(len(pred)) ]
                
            #     if(use_gpu):
            #         testing_loss += loss.cpu().data.item()
            #         class_loss_test += total_loss_class_test.cpu().data.item()
            #         crod_loss_test += total_loss_cord_test.cpu().data.item()
            #         testing_correct += torch.sum(acc[0]).cpu().item()/(len(pred)*24)
            #     else:
            #         testing_loss += loss.data.item()
            #         class_loss_test += total_loss_class_test.data.item()
            #         crod_loss_test += total_loss_cord_test.data.item()
            #         testing_correct += torch.sum(acc[0]).item()/(len(pred)*24)
                
            #     test_index += 1
        
            epoch_loss = running_loss/train_index 
            epoch_acc = running_correct/train_index * 100
            epoch_test_loss = testing_loss/test_index
            epoch_test_acc = testing_correct/test_index * 100
            
            self.history_acc.append(epoch_acc)
            self.history_loss.append(epoch_loss)
            # self.history_test_acc.append(epoch_test_acc)
            # self.history_test_loss.append(epoch_test_loss)
            self.save_parameter("./save/save_epoch_"+str(epoch)+"/","best.pkl")
            print(
                "Train Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Loss is:{:.4f}, Test Accuracy is:{:.4f} ,cost time:{}\nclass:{:.4f},cord:{:.4f},class:{:.4f},cord{:.4f}".format(
                    epoch_loss,
                    epoch_acc,
                    epoch_test_loss,
                    epoch_test_acc,
                    (datetime.datetime.now() - start_time).seconds, 
                    class_loss_train/train_index ,
                    crod_loss_train/train_index ,
                    class_loss_test/train_index ,
                    crod_loss_test/train_index ,
                )
             )
            
        self.save_parameter()
        self.save_history()

    def test(self, n_epochs, data_loader_test):

        self.model.eval()
        with torch.no_grad():
            for epoch in range(n_epochs):
                start_time = datetime.datetime.now()
                print("Epoch {}/{}".format(epoch, n_epochs))
                print("-"*10)
                self.load_parameter("./save/save_epoch_"+str(epoch)+"/"+"best.pkl")
                testing_loss = 0
                testing_correct = 0
                class_loss_test = 0
                crod_loss_test = 0
                test_index = 1
                total = len(data_loader_test)
                for data in data_loader_test:
                    # print("iter {}/{}".format(test_index, total))
                    X_train, X_pos, X_mask, x_anchor, x_class  = data
                    X_train, X_pos, X_mask, x_anchor, x_class = Variable(X_train), Variable(X_pos), Variable(X_mask), Variable(x_anchor), Variable(x_class)
                    if(use_gpu):
                        X_train = X_train.to(device)
                        X_pos = X_pos.to(device)
                        X_mask = X_mask.to(device)
                        x_anchor = x_anchor.to(device)
                        x_class = x_class.to(device)
                    
                    outputs = self.model(X_train, X_pos, X_mask)
                    
                    _,pred = torch.max(outputs['pred_logits'], 2)
                    _,pred_label = torch.max(x_class, 2)

                    x_class = x_class.float()
                    total_loss_class = 0
                    total_loss_cord = 0
                    for j in range(batch_size):
                        for i in range(self.require_number):
                            total_loss_class += self.cost_class( outputs["pred_logits"][j][i].unsqueeze(0), x_class[j][i].unsqueeze(0))
                            for i_cord in range(3):
                                total_loss_cord += self.cost_cord( outputs["pred_boxes"][j][i][i_cord], x_anchor[j][i][i_cord])
                
                    loss = total_loss_class + total_loss_cord
                    acc = [ pred_label[i] == pred[i] for i in range(len(pred)) ]
                   
                    if(use_gpu):
                        testing_loss += loss.cpu().data.item()
                        class_loss_test += total_loss_class.cpu().data.item()
                        crod_loss_test += total_loss_cord.cpu().data.item()
                        testing_correct += torch.sum(acc[0]).cpu().item()/(len(pred)*self.require_number)
                    else:
                        testing_loss += loss.data.item()
                        class_loss_test += total_loss_class.data.item()
                        crod_loss_test += total_loss_cord.data.item()
                        testing_correct += torch.sum(acc[0]).item()/(len(pred)*self.require_number)
                    
                    test_index += 1

                running_correct = 0
                running_loss = 0
                train_index = 1
                class_loss_train = 0
                crod_loss_train = 0
                epoch_loss = running_loss/train_index 
                epoch_acc = running_correct/train_index * 100
                epoch_test_loss = testing_loss/test_index
                epoch_test_acc = testing_correct/test_index * 100
                
                #self.history_acc.append(epoch_acc)
                #self.history_loss.append(epoch_loss)
                self.history_test_acc.append(epoch_test_acc)
                self.history_test_loss.append(epoch_test_loss)
                
                print(
                    "Train Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Loss is:{:.4f}, Test Accuracy is:{:.4f} ,cost time:{}\nclass:{:.4f},cord:{:.4f},class:{:.4f},cord{:.4f}".format(
                        epoch_loss,
                        epoch_acc,
                        epoch_test_loss,
                        epoch_test_acc,
                        (datetime.datetime.now() - start_time).seconds, 
                        class_loss_train/train_index ,
                        crod_loss_train/train_index ,
                        class_loss_test/train_index ,
                        crod_loss_test/train_index ,
                    )
                )
        self.save_history(save_mode= "test" )
       

    def predict(self, image):
        pass
        self.model.eval()
        if(len(image.shape) != 4):
            image = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
        output = self.model( Variable(image))
        _, preds = torch.max(output, 1)
        print(preds)
        return preds
    
    def save_history(self, file_path = './save/', save_mode = "both"):
        train_save = False
        test_save = False
        if not os.path.exists(file_path): 
            os.mkdir(file_path)
        if(save_mode == "both"):
            train_save = True
            test_save = True
        elif(save_mode == "train"):
            train_save = True
        else:
            test_save = True
            
        if(train_save):
            fo = open(file_path + "loss_history.txt", "w+")
            fo.write(str(self.history_loss))
            fo.close()
            fo = open(file_path + "acc_history.txt", "w+")
            fo.write(str(self.history_acc))
            fo.close()

        if(test_save):
            fo = open(file_path + "test_acc_history.txt", "w+")
            fo.write(str(self.history_test_acc))
            fo.close()   
            fo = open(file_path + "test_loss_history.txt", "w+")
            fo.write(str(self.history_test_loss))
            fo.close()  
        
    def save_parameter(self, file_path = './save/', file_name = None):
        if not os.path.exists(file_path): 
            os.mkdir(file_path)
        if( file_name is None):
            file_name = file_path + "model_" +str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace("-","_").replace(".","_") + ".pkl"
        else:
            file_name = file_path + file_name
        torch.save(obj=self.model.state_dict(), f=file_name)
    def load_parameter(self, file_path = './save/' ):
        # self.model.load_state_dict(torch.load('model_parameter.pkl'))
        self.model.load_state_dict(torch.load(file_path))
    def generate_mask(self, data_x, data_density, data_anchor):
        total = len(data_x)
        assert len(data_x) == len(data_density)
        
        src_list = []
        pos_list = []
        mask_list = []
        anchor_list = []
        class_list =[]
        for i in range(total):
            data_anchor_object_len = len(data_anchor[i])
            anchor_object_list = []
            class_object_list = []
            for data_anchor_object_index in range(data_anchor_object_len): 
                anchor_object_list.append(
                    [
                        data_anchor[i][data_anchor_object_index]["x_normal"], 
                        data_anchor[i][data_anchor_object_index]["y_normal"], 
                        data_anchor[i][data_anchor_object_index]["FWHM"]/10, 
                        # data_anchor[i][data_anchor_object_index]["MAG"]
                    ]
                )
                class_object_list.append([1,0])
            if(self.require_number > data_anchor_object_len):
                for j in range( self.require_number - data_anchor_object_len ):
                    anchor_object_list.append([ 0,0,0])
                    class_object_list.append([0,1])
            class_list.append(class_object_list)
            anchor_list.append( copy.deepcopy( np.array( anchor_object_list )) )
            src_list.append(cv2.resize(data_x[i], self.image_shape))
            pos = cv2.resize(data_density[i], self.image_shape)
            mask = np.ones(self.image_shape, dtype=bool)
            mask[ pos <0.02] = 0
            pos_list.append(pos)
            mask_list.append(mask)
        
        return np.array(src_list), np.array(pos_list), np.array(mask_list), np.array(anchor_list), np.array(class_list)
    

if __name__ == "__main__":
    batch_size = 1

    MYDataLoader = DataGenerate( 128, data_set_number = "L")
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

    trainer = Train(
            hidden_dim = 512,
            dropout = 0.1,
            nheads = 8,
            dim_feedforward = 2048,
            enc_layers = 6,
            dec_layers = 6,
            pre_norm = False,
            is_show = False,
            class_number = 1, 
            require_number = 10,
            image_shape = (64,64)
        )

    train_data_src, train_data_pos, train_data_mask, train_anchor, train_class = trainer.generate_mask(train_data_x, train_data_density, train_data_anchor)
    val_data_src, val_data_pos, val_data_mask, val_anchor, val_class = trainer.generate_mask(val_data_x, val_data_density, val_data_anchor)
    traindata = TensorDataset( torch.from_numpy( np.expand_dims( train_data_src ,1) ),torch.from_numpy( np.expand_dims(train_data_pos,1)),torch.from_numpy( np.expand_dims(train_data_mask,1)), torch.from_numpy( train_anchor), torch.from_numpy( train_class))
    print("data total :",len(MYDataLoader)," image shape :",val_data_src[0].shape)
    train_dataloader = DataLoader(
        dataset = traindata,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        #num_workers = 3
    )
    valdata = TensorDataset( torch.from_numpy( np.expand_dims( val_data_src,1)), torch.from_numpy( np.expand_dims(val_data_pos,1)),torch.from_numpy( np.expand_dims(val_data_mask,1)),torch.from_numpy( val_anchor), torch.from_numpy( val_class))
    val_dataloader = DataLoader(
        dataset = valdata,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        #num_workers = 3,
    )

    print(len(train_dataloader), len(val_dataloader))
    # trainer.train(100, train_dataloader, val_dataloader)
    trainer.test(100, val_dataloader)

