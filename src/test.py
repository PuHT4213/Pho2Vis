import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_processing import get_dataloader
from Pho2Vis import Pho2Vis

def batch_dealer(batch, device):
    # 图片数据
    images0 = batch['image0'].to(device)
    images1 = batch['image1'].to(device)
    # 数值特征
    aod = batch['aod'].to(device)
    rh = batch['rh'].to(device)
    month = batch['month'].to(device)
    Hour = batch['Hour'].to(device)
    # 增加一个维度，以便在通道维度上拼接
    aod = aod.unsqueeze(1)
    rh = rh.unsqueeze(1)
    month = month.unsqueeze(1)
    Hour = Hour.unsqueeze(1)

    numerical_features = torch.cat((aod, rh, month, Hour), dim=1)
    # print(Hour.shape,numerical_features.shape)
    # 可见度
    visibility = batch['visibility'].to(device).view(-1, 1)
    # 全部转换为torch.float
    images0 = images0.float()
    images1 = images1.float()
    numerical_features = numerical_features.float()
    visibility = visibility.float()

    return images0, images1, numerical_features, visibility

def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in test_loader:
            images0, images1, numerical_features, visibility = batch_dealer(batch, device)

            outputs = model(images0, images1, numerical_features)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(visibility.cpu().numpy())

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f'Test MSE: {mse:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test MAE: {mae:.4f}')
    print(f'Test R^2: {r2:.4f}')

    return mse, rmse, mae, r2

def write_results(mse, rmse, mae, r2, output_file):
    with open(output_file, 'w') as f:
        f.write(f'Test MSE: {mse:.4f}\n')
        f.write(f'Test RMSE: {rmse:.4f}\n')
        f.write(f'Test MAE: {mae:.4f}\n')
        f.write(f'Test R^2: {r2:.4f}\n')

def main():
    model_name = 'visibility_model_nofixed_1'
    model_path = './models/' + model_name + '.pth'
    output_file = './results/'+ model_name + '.txt'
    test_loader = get_dataloader('./data/test_mask_rename', './data/test_labels.csv')

    model = Pho2Vis(numerical_features=4)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    mse, rmse, mae, r2 = test(model, test_loader)
    write_results(mse, rmse, mae, r2, output_file)

if __name__ == '__main__':
    main() 

