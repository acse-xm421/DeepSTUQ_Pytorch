import torch
import torch.nn as nn
from lib.utils import enable_dropout
from lib.metrics import All_Metrics
from tqdm import tqdm 
import os
import pandas as pd
import matplotlib.pyplot as plt


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
    
    save_pred(y_true, y_pred, lower_bound, upper_bound, 'predictions.csv')
    # # Convert the tensor to a pandas DataFrame
    # df = pd.DataFrame(tensor_data.numpy())

    # # Define the CSV file path
    # csv_file_path = 'tensor_data.csv'

    # # Save the DataFrame to a CSV file
    # df.to_csv(csv_file_path, index=False, header=False)

    print("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%,  NLL: {:.4f}, \
PICP: {:.4f}%, MPIW: {:.4f}".format(mae, rmse, mape*100, nll, picp*100, mpiw))  


def save_pred(y_true, y_pred, lower_bound, upper_bound, csv_file_path):
    print(y_true.size())

    y_true_node = y_true[-1,:,-1]
    y_pred_node = y_pred[-1,:,-1]
    lower_bound_node = lower_bound[-1,:,-1]
    upper_bound_node = upper_bound[-1,:,-1]

    print(y_true_node.size())

    # y_true_tensor = y_true_node.mean(dim=0, keepdim=True)
    # y_pred_tensor = y_pred_node.mean(dim=0, keepdim=True)
    # lower_bound_tensor = lower_bound_node.mean(dim=0, keepdim=True)
    # upper_bound_tensor = upper_bound_node.mean(dim=0, keepdim=True)

    # print(y_true_tensor.size())

    # Squeeze the mean_tensor to size [1, 1, 12, 1]
    y_true_tensor_squeezed = y_true_node.squeeze()
    y_pred_tensor_squeezed = y_pred_node.squeeze()
    lower_bound_tensor_squeezed = lower_bound_node.squeeze()
    upper_bound_tensor_squeezed = upper_bound_node.squeeze()

    print(y_true_tensor_squeezed.size())

    # Create a dictionary with column names and tensors
    data = {
        'y_true': y_true_tensor_squeezed.detach().cpu().numpy(),
        'y_pred': y_pred_tensor_squeezed.detach().cpu().numpy(),
        'lower_bound': lower_bound_tensor_squeezed.detach().cpu().numpy(),
        'upper_bound': upper_bound_tensor_squeezed.detach().cpu().numpy(),
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=True)

    print("Successfully saved!")
    # In this code, we first create a dictionary where the keys are the column names, and the values are the corresponding tensors. Then, we convert this dictionary into a pandas DataFrame. Finally, we save the DataFrame to a CSV file with the specified column names.

def plot_vi(csv_file_path, png_file_path):
    # Sample data
    df = pd.read_csv(csv_file_path)
    # Create a DataFrame from the data
    # df = pd.DataFrame(data)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['y_true'], color='black', label='y_true')
    plt.plot(df.index, df['y_pred'], color='blue', label='y_pred')
    plt.fill_between(df.index, df['lower_bound'], df['upper_bound'], color='lightgray', alpha=0.5, label='Prediction Range')

    # Add labels and legend
    plt.xlabel('Indices')
    plt.ylabel('Values')
    plt.legend()

    # Show the plot
    plt.title('Plot of y_true, y_pred, and Prediction Range')
    plt.grid(True)
    plt.savefig(png_file_path)
    
    plt.show()





