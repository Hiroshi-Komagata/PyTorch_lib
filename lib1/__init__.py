# %% [markdown]
# # 共通処理

# %%
# ReadMe
README = 'Common Library for PyTorch\nAuthor: H. Hiroshi\nVer:1.0.1'

# %% [markdown]
# ## ライブラリ

# %%
# 必要ライブラリのインポート

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import display

# %%
# torch関連ライブラリのインポート

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# %% [markdown]
# ## 損失計算

# %%
# 損失計算用
def eval_loss(loader, device, net, criterion):
    """
    損失率を計算する。
    
    Parameters
    ----------
    loade  : データローダー
    device  : 処理デバイス
    net  : 学習対象のモデルインスタンス
    criterion  : 損失関数のインスタンス

    Returns
    -------
    loss : 損失計算結果
    """

    # データローダーから最初の1セットを取得する
    for images, labels in loader:
        break

    # デバイスの割り当て
    inputs = images.to(device)
    labels = labels.to(device)

    # 予測計算
    outputs = net(inputs)

    #  損失計算
    loss = criterion(outputs, labels)

    return loss

# %% [markdown]
# ## モデルを保存する

# %%
import datetime
import os

def SaveModel(model, save_path, file_name_pram=None):
    now = datetime.datetime.now()
    # 学習済みの重みパラメータを保存する
    if file_name_pram is not None:
        save_path = os.path.join(save_path, now.strftime("%Y%m%d%H%M%S")  + '_' + file_name_pram + '.pth')
    else:
        save_path = os.path.join(save_path, now.strftime("%Y%m%d%H%M%S") + '.pth')
    torch.save(model, save_path)

# %% [markdown]
# ## 学習処理

# %%
# 学習用関数
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history, scheduler = None, save_better_only=False, save_path=None):
    """
    学習処理を実施する。
    
    Parameters
    ----------
    net  : 学習対象のモデルインスタンス
    optimizer  : 最適化関数のインスタンス
    criterion  : 損失関数のインスタンス
    num_epochs   : 繰り返し数
    train_loader  : 訓練用のデータローダー
    test_loader  : 検証用のデータローダー
    device  : 処理デバイス
    history  : 学習結果（繰り返し数、訓練損失、訓練精度、検証損失、検証精度）
    scheduler  : スケジューラー
    save_better_only  : Trueの場合、損失率が最小の場合、保存する
    save_path  : モデルを保存するパス

    Returns
    -------
    history  : 学習結果（繰り返し数、訓練損失、訓練精度、検証損失、検証精度）
    """

    # tqdmライブラリのインポート
    from tqdm.notebook import tqdm

    base_epochs = len(history)
  
    for epoch in range(base_epochs, num_epochs+base_epochs):
        # 1エポックあたりの正解数(精度計算用)
        n_train_acc, n_val_acc = 0, 0
        # 1エポックあたりの累積損失(平均化前)
        train_loss, val_loss = 0, 0
        # 1エポックあたりのデータ累積件数
        n_train, n_test = 0, 0

        #訓練フェーズ
        net.train()

        for inputs, labels in tqdm(train_loader):
            # 1バッチあたりのデータ件数
            train_batch_size = len(labels)
            # 1エポックあたりのデータ累積件数
            n_train += train_batch_size
    
            # デバイスの割り当て
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # 予測ラベル導出
            predicted = torch.max(outputs, 1)[1]

            # 平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            train_loss += loss.item() * train_batch_size 
            n_train_acc += (predicted == labels).sum().item() 

        #予測フェーズ
        net.eval()

        for inputs_test, labels_test in test_loader:
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels_test)
            # 1エポックあたりのデータ累積件数
            n_test += test_batch_size

            # デバイスの割り当て
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # 予測計算
            outputs_test = net(inputs_test)

            # 損失計算
            loss_test = criterion(outputs_test, labels_test)
 
            # 予測ラベル導出
            predicted_test = torch.max(outputs_test, 1)[1]

            #  平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss +=  loss_test.item() * test_batch_size
            n_val_acc +=  (predicted_test == labels_test).sum().item()

        if scheduler != None:
            # スケジューラを更新する
            scheduler.step()
        
        # 精度計算
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test
        # 結果表示
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}')
        # 記録
        item = np.array([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])

        # モデルを保存する
        if save_path is not None:
            if save_better_only and epoch > 1 and history[:, 3].min() < avg_val_loss:
                pass
            else:
                SaveModel(net, save_path, 'epoch{}_valloss{:.5f}_valacc{:.5f}'.format(epoch+1, avg_val_loss, val_acc))

        history = np.vstack((history, item))

    return history

# %% [markdown]
# ##  学習ログ解析用

# %%
# 学習ログ解析

def evaluate_history(history):
    """
    学習曲線を表示する。
    
    Parameters
    ----------
    history  : 学習結果（繰り返し数、訓練損失、訓練精度、検証損失、検証精度）
    """

    #損失と精度の確認
    print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}') 
    print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='訓練')
    plt.plot(history[:,0], history[:,3], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('損失')
    plt.title('学習曲線(損失)')
    plt.legend()
    plt.show()

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='訓練')
    plt.plot(history[:,0], history[:,4], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('精度')
    plt.title('学習曲線(精度)')
    plt.legend()
    plt.show()

# %% [markdown]
# ## イメージとラベルの表示

# %%
# イメージとラベル表示
def show_images_labels(loader, classes, net, device):
    """
    学習処理を実施する。
    
    Parameters
    ----------
    loader  : 検証用データローダー
    classes  : 正解データに対応するラベル値のリスト
    net  : 学習対象のモデルインスタンス, Noneの場合、正解データのみ表示する
    device  : 処理デバイス

    """

    # データローダーから最初の1セットを取得する
    for images, labels in loader:
        break
    # 表示数は50個とバッチサイズのうち小さい方
    n_size = min(len(images), 50)

    if net is not None:
      # デバイスの割り当て
      inputs = images.to(device)
      labels = labels.to(device)

      # 予測計算
      outputs = net(inputs)
      predicted = torch.max(outputs,1)[1]

    # 最初のn_size個の表示
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # netがNoneでない場合は、予測結果もタイトルに表示する
        if net is not None:
          predicted_name = classes[predicted[i]]
          # 正解かどうかで色分けをする
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # netがNoneの場合は、正解ラベルのみ表示
        else:
          ax.set_title(label_name, fontsize=20)
        # TensorをNumPyに変換
        image_np = images[i].numpy().copy()
        # 軸の順番変更 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 値の範囲を[-1, 1] -> [0, 1]に戻す
        img = (img + 1)/2
        # 結果表示
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()


# %% [markdown]
# ## 乱数初期化

# %%
def torch_seed(seed=123):
    """
    PyTorch乱数固定用
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# %% [markdown]
# ## モデルを使用して画像を分類する

# %%
def ImageRec(model, input_image, classes, device):
    """
    モデルを使用して画像を分類する。
    
    Parameters
    ----------
    model  : モデルインスタンス
    input_image  : 判定画像
    classes  : 分類
    device  : 処理デバイス
    """
    
    # 前処理の定義
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # 前処理を行う
    input_tensor = preprocess(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    input_tensor_normalize = transforms.Normalize(0.5, 0.5)(input_tensor)

    # 推論を行う
    model.eval()
    with torch.no_grad():
        output = model(input_tensor_normalize)

    # クラス分類結果を取得
    predicted = F.softmax(output[0], dim=0)

    # 結果を表示する
    top5_probs, top5_idxs = torch.topk(predicted, 3)

    for i in range(3):
        print(f'{classes[top5_idxs[i]]}: {top5_probs[i]:.2%}')
    
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    input_tensor = input_tensor.to('cpu')

    # 結果表示
    image = transforms.ToPILImage()(input_tensor[0])

    _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.axis('off')
    plt.show()


