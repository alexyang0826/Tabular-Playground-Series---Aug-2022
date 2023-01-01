# Tabular-Playground-Series---Aug-2022

### 1. Specification of dependencies

using .ipynb

change paths to your `model.pt` , `train.csv` and `test.csv` in *Path* block

    MODEL_PATH = './models/model.pt'
    TRAIN_PATH = './train.csv'
    TEST_PATH = './test.csv'

### 2. Training code

#### Hyperparameters :

    train_val_ratio : 0.9999  # split train and val
    epoch = 429 # epoch numbers
    batch_size = 156 # batch size
    save_best = False   # save hightest val accuracy or not
#### Model:
Three layers nn:

input_shape > 32 > 64 > 1

optimizer: Adam( lr=0.001, betas=( 0.9, 0.999), eps=1e-08 )

    class Model(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x)
### 3. Evaluation code

Predict result and save result to `submission_p.csv`

    test_data = test_df_clean.to_numpy()
    test_ds = TaskDataset(test_data, return_y=False)
    print("test num: ", test_ds.__len__())
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=0,
        drop_last=False,
        shuffle=False)

    model.eval()
    pred = []
    for x in tqdm(test_dl):
        x = x.to(device)
        y_pred = model(x)
        output = torch.sigmoid(y_pred)
        output = output.cpu().detach().numpy()
        for i in range(len(output)):
            pred.append(output[i][0])
    result = pd.DataFrame({'id': test_df['id'], 'failure': pred})
    result.to_csv('submission_p.csv', index=0)
    result

### 4. Pre-trained models
This is my mode with private score **0.5928**

[Download Link](https://1drv.ms/u/s!Am-Jv_BeNcKfi7AsJX8IrjYvmqZ0-w?e=A8XNgM)

### 5. Result
Private score **0.5928**
![Result](/result/result.png)