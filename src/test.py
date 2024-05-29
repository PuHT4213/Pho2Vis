import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from train import batch_dealer
from data_processing import get_dataloader
from Pho2Vis import Pho2Vis

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
    output_file = './results'+ model_name + '.txt'
    test_loader = get_dataloader('./new_May_25/test_mask_rename', './new_May_25/test_labels.csv')

    model = Pho2Vis(numerical_features=4)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    mse, rmse, mae, r2 = test(model, test_loader)
    write_results(mse, rmse, mae, r2, output_file)

if __name__ == '__main__':
    main() 

