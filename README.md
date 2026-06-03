LENS_ADNet/
  ├── data/
  │   ├── download_adni.py
  │   └── preprocess.py
  ├── model/
  │   ├── cnn_extractor.py Step 1
  │   ├── transformer_block.py Step 2
  │   ├── fusion.py Step 3
  │   ├── symbolic_layer.py Step 4 ✦
  │   └── lens_adnet.py Step 5
  ├── train/
  │   ├── train.py Step 6
  │   └── evaluate.py Step 7
  ├── explainability/
  │   └── explain.py Step 8
  ├── outputs/  (auto-created — saved models, plots)
  ├── requirements.txt First
  └── README.md