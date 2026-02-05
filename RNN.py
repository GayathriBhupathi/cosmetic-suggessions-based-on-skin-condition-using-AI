import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output from the last time step

        return out

idx_to_classes = {0: 'Apple___Apple_scab',
                  1: 'Apple___Black_rot',
                  2: 'Apple___Cedar_apple_rust',
                  3: 'Apple___healthy',
                  4: 'Background_without_leaves',
                  5: 'Blueberry___healthy',
                  6: 'Cherry___Powdery_mildew',
                  7: 'Cherry___healthy',
                  8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                  9: 'Corn___Common_rust',
                  10: 'Corn___Northern_Leaf_Blight',
                  11: 'Corn___healthy',
                  12: 'Grape___Black_rot',
                  13: 'Grape___Esca_(Black_Measles)',
                  14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                  15: 'Grape___healthy',
                  16: 'Orange___Haunglongbing_(Citrus_greening)',
                  17: 'Peach___Bacterial_spot',
                  18: 'Peach___healthy',
                  19: 'Pepper,_bell___Bacterial_spot',
                  20: 'Pepper,_bell___healthy',
                  21: 'Potato___Early_blight',
                  22: 'Potato___Late_blight',
                  23: 'Potato___healthy',
                  24: 'Raspberry___healthy',
                  25: 'Soybean___healthy',
                  26: 'Squash___Powdery_mildew',
                  27: 'Strawberry___Leaf_scorch',
                  28: 'Strawberry___healthy',
                  29: 'Tomato___Bacterial_spot',
                  30: 'Tomato___Early_blight',
                  31: 'Tomato___Late_blight',
                  32: 'Tomato___Leaf_Mold',
                  33: 'Tomato___Septoria_leaf_spot',
                  34: 'Tomato___Spider_mites Two-spotted_spider_mite',
                  35: 'Tomato___Target_Spot',
                  36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                  37: 'Tomato___Tomato_mosaic_virus',
                  38: 'Tomato___healthy'}



# Assuming input_size is appropriate for your data
input_size = 3  # Update according to your input size
hidden_size = 64
num_layers = 2
num_classes = len(idx_to_classes)

rnn_model = RNN(input_size, hidden_size, num_layers, num_classes)
