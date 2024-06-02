import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from Pho2Vis import Pho2Vis
from data_processing import get_dataloader, get_scaler
from test import test, write_results
import os
# set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


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

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 创建目录
    if not os.path.exists('.//runs'):
        os.makedirs('.//runs')
    writer = SummaryWriter('.//runs//visibility_experiment')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images0, images1, numerical_features, visibility = batch_dealer(batch, device)

            optimizer.zero_grad()
            outputs = model(images0, images1, numerical_features)
            loss = criterion(outputs, visibility)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images0.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Training Loss', epoch_loss, epoch)

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images0, images1, numerical_features, visibility = batch_dealer(batch, device)

                    outputs = model(images0, images1, numerical_features)
                    loss = criterion(outputs, visibility)
                    val_loss += loss.item() * images0.size(0)

            val_epoch_loss = val_loss / len(val_loader.dataset)
            writer.add_scalar('Validation Loss', val_epoch_loss, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}')

    writer.close()
    return model

def main():
    # 获取当前目录
    train_file = './data/train_labels.csv'
    test_file = './data/test_labels.csv'
    train_image_dir = './data/train_mask_rename'
    test_image_dir = './data/test_mask_rename'
    output_file = './models/tmp_visibility_model.pth'
    batch_size = 16
    num_epochs = 60
    numerical_features = 4
    target_size = (299, 299)

    train_loader = get_dataloader(train_image_dir, train_file, batch_size,target_size=target_size)
    test_loader = get_dataloader(test_image_dir, test_file, batch_size,target_size=target_size)

    model = Pho2Vis(numerical_features=numerical_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    trained_model = train(model, train_loader, test_loader, optimizer, criterion, num_epochs)

    # 测试模型
    mse, rmse, mae, r2 = test(trained_model, test_loader)
    write_results(mse, rmse, mae, r2, './results/tmp_results.txt')

    # Save the model
    torch.save(trained_model.state_dict(), output_file)

if __name__ == "__main__":
    main()