import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from model import build_model, count_params
from train import prepare_model_for_qat, make_loaders, evaluate

def train_optimal(data_root):

    search_space = [
        {"base_c": 8, "img": 64, "se": False, "wave": 1},
        {"base_c": 16, "img": 96, "se": False, "wave": 1},
        {"base_c": 16, "img": 128, "se": True, "wave": 2},
        {"base_c": 24, "img": 96, "se": True, "wave": 1},
        {"base_c": 32, "img": 128, "se": True, "wave": 2},
    ]

    best_score = -1
    best_model = None
    best_cfg = None

    for cfg in search_space:
        print("\nTesting config:", cfg)

        train_loader, val_loader, num_classes = make_loaders(
            data_root, img_size=cfg["img"], batch_size=64
        )

        model = build_model(
            num_classes,
            base_c=cfg["base_c"],
            use_se=cfg["se"],
            wavelets=cfg["wave"]
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        model = prepare_model_for_qat(model)

        opt = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):  # quick search training
            model.train()
            for x,y in train_loader:
                x,y = x.cuda(), y.cuda()
                opt.zero_grad()
                out = model(x)
                loss = criterion(out,y)
                loss.backward()
                opt.step()

        val_loss, val_acc = evaluate(model, val_loader, "cuda")

        params = count_params(model)
        score = val_acc - 0.000001 * params

        print("Val acc:", val_acc, "Params:", params, "Score:", score)

        if score > best_score:
            best_score = score
            best_model = copy.deepcopy(model)
            best_cfg = cfg

    print("\nBest config:", best_cfg)

    torch.save(best_model.state_dict(), "best_optimal_model.pth")
