import torch
import torch.nn as nn
from lib.utils import enable_dropout
from lib.metrics import All_Metrics
from tqdm import tqdm 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import csv
import pickle


####======MC+Heter========####
def combined_test(model,num_samples,args, data_loader, scaler, T=torch.zeros(1).cuda(), logger=None, path=None):#
    model.eval()
    enable_dropout(model)
    nll_fun = nn.GaussianNLLLoss()
    y_true = []
    with torch.no_grad():
        for batch_idx, (_, target) in enumerate(data_loader):
            label = target[..., :args.output_dim]
            y_true.append(label)
    y_true = scaler.inverse_transform(torch.cat(y_true, dim=0)).squeeze(3)
    
    mc_mus = torch.empty(0, y_true.size(0), y_true.size(1), y_true.size(2)).cuda()
    mc_log_vars = torch.empty(0, y_true.size(0),y_true.size(1), y_true.size(2)).cuda()
    
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            mu_pred = []
            log_var_pred = []
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                mu, log_var = model.forward(data, target, teacher_forcing_ratio=0)
                #print(mu.size())
                mu_pred.append(mu.squeeze(3))
                log_var_pred.append(log_var.squeeze(3))
        
            if args.real_value:
                mu_pred = torch.cat(mu_pred, dim=0)
            else:
                mu_pred = scaler.inverse_transform(torch.cat(mu_pred, dim=0))     
            log_var_pred = torch.cat(log_var_pred, dim=0)    

            # print(mc_mus.size(),mu_pred.size())    
            mc_mus = torch.vstack((mc_mus,mu_pred.unsqueeze(0)))  
            # print(mc_mus.size())    
            mc_log_vars = torch.vstack((mc_log_vars,log_var_pred.unsqueeze(0))) 
    
    temperature = torch.exp(T)   
    print(temperature)  
    y_pred = torch.mean(mc_mus, axis=0)
    total_var = torch.var(mc_mus, axis=0)+torch.exp(torch.mean(mc_log_vars, axis=0))/temperature   
    total_std = total_var**0.5 
    
    mpiw = 2*1.96*torch.mean(total_std)    
    nll = nll_fun(y_pred.ravel(), y_true.ravel(), total_var.ravel())

    # Calculate metrics
    mae = torch.abs(y_pred - y_true).mean().item()
    rmse = torch.sqrt(((y_pred - y_true) ** 2).mean()).item()
    mape = (torch.abs(y_pred - y_true) / (y_true + 1e-6)).mean().item()

    lower_bound = y_pred-1.96*total_std
    upper_bound = y_pred+1.96*total_std
    in_num = torch.sum((y_true >= lower_bound)&(y_true <= upper_bound ))
    picp = in_num/(y_true.size(0)*y_true.size(1)*y_true.size(2))
    
    save_pred(y_true, y_pred, lower_bound, upper_bound)
    # # Convert the tensor to a pandas DataFrame
    # df = pd.DataFrame(tensor_data.numpy())

    # # Define the CSV file path
    # csv_file_path = 'tensor_data.csv'

    # # Save the DataFrame to a CSV file
    # df.to_csv(csv_file_path, index=False, header=False)

    print("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%,  NLL: {:.4f}, \
PICP: {:.4f}%, MPIW: {:.4f}".format(mae, rmse, mape*100, nll, picp*100, mpiw))  

def extract_from_window(data, single=False):#
    B, H, N = data.shape
    extracted_data = []

    if single:
        for i in range(B-H+1):
            extracted_data.append(data[i, -1, :].reshape((1,N)))
    else:
        for i in range(B-H+1):
            if i==0:
                extracted_data.append(data[i,:,:].reshape((H,N)))
            else:
                extracted_data.append(data[i, -1, :].reshape((1,N)))
    
    concatenated_data = torch.cat(extracted_data, axis=0)

    return concatenated_data

def save_pred(y_true, y_pred, lower_bound, upper_bound):
    print("y_true", y_true.size())

    y_true_extracted_tensor = extract_from_window(y_true)
    y_pred_extracted_tensor = extract_from_window(y_pred)
    lower_bound_extracted_tensor = extract_from_window(lower_bound)
    upper_bound_extracted_tensor = extract_from_window(upper_bound)

    print("extract", y_true_extracted_tensor.size())

    # Create a dictionary with column names and tensors
    data = {
        'y_true': y_true_extracted_tensor.detach().cpu().numpy(),
        'y_pred': y_pred_extracted_tensor.detach().cpu().numpy(),
        'lower_bound': lower_bound_extracted_tensor.detach().cpu().numpy(),
        'upper_bound': upper_bound_extracted_tensor.detach().cpu().numpy(),
    }

    # Specify the file path where you want to save the data
    file_path = 'data.pkl'

    # Open the file in binary write mode and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

    # Save the DataFrame to a CSV file
    # df.to_csv(csv_file_path, index=True)

    print("Successfully saved!")

def plot_vi(file_path, png_file_path):
    # Sample data
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    print(loaded_data)
    print(loaded_data.shape)

    y_true = loaded_data['y_true']
    y_pred = loaded_data['y_pred']
    lower_bound = loaded_data['lower_bound']
    upper_bound = loaded_data['upper_bound']

    y_true = np.transpose(y_true, (1,0))
    y_pred = np.transpose(y_pred, (1,0))
    lower_bound = np.transpose(lower_bound, (1,0))
    upper_bound = np.transpose(upper_bound, (1,0))
    # zip y_true, y_pred, lower_bound, upper_bound
    # give me an array

    for idx, y_true,y_pred,lower_bound,upper_bound in enumerate(zip(y_true, y_pred, lower_bound, upper_bound)):
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(np.range(len(y_true)), y_true, color='black', label='y_true')
        plt.plot(np.range(len(y_pred)), y_pred, color='blue', label='y_pred')
        plt.fill_between(np.range(len(lower_bound)), lower_bound, upper_bound, color='lightgray', alpha=0.5, label='Prediction Range')

        # Add labels and legend
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()

        # Show the plot
        plt.title('Variational Inference plot')
        plt.grid(True)
        png_file_path = "idx-" + png_file_path
        plt.savefig(png_file_path)
        
        plt.show()
