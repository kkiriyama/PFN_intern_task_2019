## 実行手順

`datasets` ディレクトリを `src` ディレクトリと同じ階層に置いてください

### 課題1

```bash
python task_1.py
```

### 課題2

```bash
python task_2.py
```

### 課題3

```bash
// SGDで学習する場合
python task_3.py SGD

// momentum SGDで学習する場合
python task_3.py momentumSGD
```

## 課題4

```bash
// Adamでの学習 model.pklが生成されます
python task_4.py train
```

```bash
// テストデータセットでの予測 モデルのpathを指定します
python task_4.py test model.pkl
```
