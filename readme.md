# Gossip Distillation

Gossip Learning(PULL) × Knowledge Distillation

# 使い方

## 必要なもの

- データセット
  - 各デバイスにしか見えないデータセット(Private Dataset)
  - 通信用のデータセット(Distillation Dataset)
- モデル
  - args.diffrent_model
    - True...Resnet18, mobilenetv3とかを使う(予定)
    - False...Resnet18のみ

## 使い方
1. デバイス$n$個でPrivate Datasetを用いて学習させる
2. Distillation Datasetを用いて推論をし、結果を所持しておく
3. (for every device as device A)
   1. select another device randomly(=device B)
   2. Download the results of Device B's inference to Device A and Integrate data.
   3. Create a new dataset with the integrated data as labels and retrain the model using it.
   4. Infer the Distillation Dataset with the model after training and retain the results.

# 注意点
- Dataset→Dataloaderにおいてshuffle=Falseにする(Distillation Datasetで推論するときに順番が狂う)

# Todo
- 様々な種類のモデルを組み合わせた実験
- 非同期で3.の処理を行う(時短)
- 結果のビジュアライズ
- モジュール化