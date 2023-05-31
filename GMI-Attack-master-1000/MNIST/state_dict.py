import torch, os, utils, dataloader, torchvision, classify
import torch.nn as nn
import torch.optim as optim
from utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file = "./MNIST.json"
args =  load_json(json_file = file)

#transform = transforms.Compose([
#	transforms.Resize((32,32)),
#	transforms.ToTensor(),
#])

train_file_path = args['dataset']['train_file_path']
test_file_path = args['dataset']['test_file_path']

train_loader = init_dataloader(args, train_file_path, 64, mode = "not")
test_loader = init_dataloader(args, test_file_path, 64, mode = "not")

lr = 0.01
momentum = 0.9
weight_decay = 0.0001
batch_size = 64
num_epochs = 10

model = classify.MCNN(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 입력 데이터를 device로 보냅니다
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward 계산
        _, outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward 계산 및 Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if (i+1) % 100 == 0:
            print('[MCNN] Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# 모델의 dict_state를 저장합니다
torch.save({'state_dict':model.state_dict()}, os.path.join("./", 'mcnn_dict_state.tar'))

# Initialize MCNN model
model.load_state_dict(torch.load('mcnn_dict_state.tar')['state_dict'])

# Set model to evaluation mode
model.eval()

# Test the model on the test dataset
correct = 0
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    _, output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()
print('[MCNN] Test set Accuracy : {:.2f}%'.format(100. * correct / len(test_loader.dataset)))


