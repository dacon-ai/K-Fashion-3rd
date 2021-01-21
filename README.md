# k-fashion-instance-segmentation

## 라이브러리 설치

### Conda (Python 3.8)

```shell script
$ conda install pytorch==1.6.0 torchvision cudatoolkit=10.2 -c pytorch -y
$ conda install pandas
```


### mmdetection

```
$ pip install mmcv-full==latest+torch1.6.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
$ cd mmdetection
$ pip install -r requirements/build.txt
$ pip install --no-cache-dir -e .
```


## 학습 데이터 생성
### Dataset-1
```shell script
$ python tools/split_dataset.py \
    --input-json  /data/train.json \
    --input-csv   /data/train.csv \
    --val-ratio   0.1 \
    --output-dir  /data/
```

### Dataset-2
```shell script
$ python tools/split_dataset.py \
    --input-json  /data/train.json \
    --input-csv   /data/train.csv \
    --val-ratio   0.01 \
    --output-dir  /data/
```


## 학습
### Model-1
```shell script
$ cd final/model-1
$ GPUS=8 bash run_train.sh
```

### Model-2
```shell script
$ cd final/model-2
$ GPUS=8 bash run_train.sh
```

### Model-2.1
* Single Node
```shell script
$ cd final/model-2.1
$ GPUS=8 bash run_train.sh
```

* Distributed 
```shell script
$ cd final/model-2.1
$ NUM_NODE=4 $NODE_RANK=0 GPUS=8 bash run_train_distributed.sh  # node-0
$ NUM_NODE=4 $NODE_RANK=1 GPUS=8 bash run_train_distributed.sh  # node-1
$ NUM_NODE=4 $NODE_RANK=2 GPUS=8 bash run_train_distributed.sh  # node-2
$ NUM_NODE=4 $NODE_RANK=3 GPUS=8 bash run_train_distributed.sh  # node-3
```


## 앙상블 테스트 및 제출파일 생성
### Model-1 + Model-2 (리더보드)
```shell script
$ cd final/submission
$ GPUS=8 bash run_test_ensemble2.sh
...
$ head leaderboard_ensemble2_thr0.5_segm.csv
```

### Model-1 + Model-2.1 (최종 제출)
```shell script
$ cd final/submission
$ GPUS=8 bash run_test_ensemble2_1.sh
...
$ head final_ensemble2_1_thr0.5_segm.csv
```

