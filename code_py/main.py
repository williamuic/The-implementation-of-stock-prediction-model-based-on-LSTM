import pandas as pd
import os
import sys
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Network(Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden

def train(config, logger, train_and_valid_data):

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()     # transform to a Tensor
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)    # DataLoader automatically generates trainable Batch data

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu") # Choose CPU training or GPU training
    model = Network(config).to(device)      # If it's GPU training, .to(device) will copy the model/data to GPU video memory
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # define optimizer
    criterion = torch.nn.MSELoss()      # define loss

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch+1, config.epoch))
        model.train()
        train_loss_array = []
        hidden_train = None
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            optimizer.zero_grad()               # The gradient should be set to 0 before training
            pred_Y, hidden_train = model(_train_X, hidden_train)    # Forward calculation
            hidden_train = None             # reset hidden
            loss = criterion(pred_Y, _train_Y)  # calculate loss
            loss.backward()                     # backpropagation of loss
            optimizer.step()                    # Update parameters with the optimizer
            train_loss_array.append(loss.item())
            global_step += 1

        # The following is the early stop mechanism. When the model training failed to improve the prediction effect of
        # the validation set by continuous config.patience number of epoch, it stopped to prevent overfitting.
        model.eval()
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, hidden_valid = model(_valid_X, hidden_valid)
            hidden_valid = None
            loss = criterion(pred_Y, _valid_Y)  # only forward calculation, no back propagation process
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
              "The valid loss is {:.6f}.".format(valid_loss_cur))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + "model")  # save the model
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience: # If the validation set indicators do not
                                        # improve for consecutive pacience number of epoch, the training is stopped
                logger.info(" The training stops early in epoch {}".format(epoch))
                break

def predict(config, test_X):
    #  Get the test data
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # load the model
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = Network(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + "model"))   # load model parameters

    # Firstly define a tensor to hold the prediction results
    result = torch.Tensor().to(device)

    # prediction process
    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X, hidden_predict = model(data_X, hidden_predict)
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()    # the numpy format data is returned

class Config:
    # parameters
    feature_columns = list(range(1,7))     # The index of columns that will be treated as feature, no 0 because
                                            # the first column is date
    label_columns = [4,5]                  # The index of columns that will be predicted, we predict the high and low
                                                        # value in the same time
    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)  # feature doesn't have to start at 0

    predict_day = 1            # How many days will be predicted

    # network parameters
    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 128           # The size of the hidden layer of the LSTM,  also the size of the output
    lstm_layers = 2             # Number of stacked layers in the LSTM
    dropout_rate = 0.2          # probability of dropout
    time_step = 20       # This parameter sets the number of the previous days' data will be used to do the prediction,
                    # and it is also the number of time steps of LSTM. Please ensure that the amount of training data is greater than it

    # training parameters
    use_cuda = False            # whether to use GPU training or not

    train_data_rate = 0.95      # The proportion of the original data that will be used as training data, the proportion test data is 1-train_data_rate
    valid_data_rate = 0.15      # The proportion of the original data that will be used as validation data

    batch_size = 128
    learning_rate = 0.001
    epoch = 50                  # The number of training times
    patience = 5                # The number of training times before an early stop
    random_seed = 42            # Random seed, guaranteed to be repeatable

    # paths
    data_path = "C:/Users/18430/PycharmProjects/Financial/data/data.csv"
    model_save_path = "./checkpoint/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # create the directory recursively

class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        self.mean = np.mean(self.data, axis=0)              # mean of the data
        self.std = np.std(self.data, axis=0)                # standard deviation of the data
        self.norm_data = (self.data - self.mean)/self.std   # normalization

        self.start_num_in_test = 0

    def read_data(self):                # read the original data
        init_data = pd.read_csv(self.config.data_path, usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()     # .columns.tolist() is to get the column name

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,
                                    self.config.label_in_feature_index]
        #For every example, the label is the data of next few days


        # Every time_step number of row will be used as a sample, and the two samples will be staggered on one line, for example: rows 1-20, rows 2-21...
        train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]


        train_x, train_y = np.array(train_x), np.array(train_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,random_state=self.config.random_seed,
                                                              shuffle=True)   # Split the training and validation sets, and shuffle the
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)     # Prevent time_steps from exceeding the number of test sets
        self.start_num_in_test = feature_data.shape[0] % sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        # Every time_step number of row will be used as a sample, and the two samples will be staggered on one line, for example: rows 1-20, rows 2-21...
        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        if return_label_data:       # The test set  has no label
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)

def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    # Print the log to screen
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : ,
                                            config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]   # Restore the data with the saved mean and variance
    assert label_data.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    # Calculation of loss
    # The following are 2 kinds of  calculation of loss. The result is the same.
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss =np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day] ) ** 2, axis=0)
    loss_norm = loss/(origin_data.std[config.label_in_feature_index] ** 2)
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [ x + config.predict_day for x in label_X]

    for i in range(label_column_num):
        plt.figure(i+1)                     #Plot the prediction data
        plt.plot(label_X, label_data[:, i], label='Original')
        plt.plot(predict_X, predict_data[:, i], label='Predicted')
        plt.title("Predict the {} price".format(label_name[i]))
        plt.legend()
        logger.info("The predicted  {} price for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
                str(np.squeeze(predict_data[-config.predict_day:, i])))
    plt.show()

def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)
        data_gainer = Data(config)

        train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
        train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
        pred_result = predict(config, test_X)
        draw(config, data_gainer, logger, pred_result)
    except Exception:
        logger.error("Run Error", exc_info=True)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    con = Config()
    for key in dir(args):  # The  dir(args) function gets all the attributes of args
        if not key.startswith("_"):  # Remove attributes such as __name__ from args
            setattr(con, key, getattr(args, key))  # Assign the attribute value to Config

    main(con)
