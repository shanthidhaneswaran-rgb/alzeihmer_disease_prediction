'''

import sys
import os
import torch

# 1. Fix the path to the root folder
root_folder = r'F:\Program\LENS_ADNet'
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)

# 2. Import the correct class name from your file
from model.lens_adnet import LENSADNet 

def main():
    weights_path = 'lens_adnet_best.pt'
    
    # FIX 1: Change n_clinical to 3 (The error says it expects 259, which is 128+128+3)
    model = LENSADNet(n_clinical=3, n_classes=3) 
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # FIX 2: Use strict=False
        # This allows the model to load even if the CNN layer names don't match perfectly.
        # NOTE: The CNN weights might stay random, but the Transformer/Symbolic/Classifier will load.
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        
        print("✅ Weights loaded with some architectural differences.")
        if unexpected:
            print(f"⚠️ Warning: Found extra layers in .pt file: {len(unexpected)}")
    else:
        print(f"❌ Error: {weights_path} not found.")
        return

    model.eval()

    # Match the 3D expectations: [Batch, Channel, Depth, Height, Width]
    # Changed Channels from 3 to 1 to fix the 'expected 1 channels' error
    dummy_mri = torch.randn(1, 1, 16, 224, 224) 
    
    # Clinical data
    dummy_clinical = torch.randn(1, 3) 

    with torch.no_grad():
        try:
            # We use *args because the model might return multiple values
            outputs = model(dummy_mri, dummy_clinical)
            
            # Since your forward returns (logits, rs, rw, attn_w)
            logits = outputs[0] 
            
            print("-" * 30)
            print("🚀 Model reached the end of the calculation!")
            print(f"Output Raw Logits: {logits}")
            
            # Get the prediction
            _, predicted = torch.max(logits, 1)
            names = {0: "CN", 1: "MCI", 2: "AD"}
            print(f"Predicted Category: {names[predicted.item()]}")
            
        except Exception as e:
            print(f"❌ Execution failed: {e}")


if __name__ == "__main__":
    main()


'''

import sys
import os
import torch

# 1. Fix the path to the root folder
root_folder = r'F:\Program\LENS_ADNet'
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)

from model.lens_adnet import LENSADNet 

def main():
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = 'lens_adnet_best.pt'
    
    # Initialize model
    # Ensure n_clinical matches the features used during training
    model = LENSADNet(n_clinical=3, n_classes=3) 
    
    if os.path.exists(weights_path):
        # weights_only=True is safer for modern PyTorch if the file allows it
        checkpoint = torch.load(weights_path, map_location=device)
        
        # If checkpoint is a dict containing 'state_dict', extract it
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # strict=False is a "safety net" but be wary if major layers are missing
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"✅ Weights loaded on {device}")
        if missing:
            print(f"⚠️ Missing keys (using random init): {len(missing)}")
    else:
        print(f"❌ Error: {weights_path} not found.")
        return

    model.to(device)
    model.eval()

    # --- INPUT PREPARATION ---
    # MRI: [Batch, Channel, Depth, Height, Width]
    # Standard LENS-ADNet usually expects float32
    dummy_mri = torch.randn(1, 1, 16, 224, 224).to(device).float()
    
    # Clinical: [Batch, n_features]
    dummy_clinical = torch.randn(1, 3).to(device).float() 

    # --- INFERENCE ---
    with torch.no_grad():
        try:
            # LENS-ADNet typically returns a tuple: (logits, rs, rw, attn)
            outputs = model(dummy_mri, dummy_clinical)
            
            # Robustly handle output unpacking
            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs
            
            print("-" * 30)
            print("🚀 Inference Successful!")
            
            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)
            conf, predicted = torch.max(probs, 1)
            
            names = {0: "CN", 1: "MCI", 2: "AD"}
            pred_idx = predicted.item()
            
            print(f"Predicted Category: {names[pred_idx]}")
            print(f"Confidence: {conf.item():.2%}")
            print(f"Raw Logits: {logits.cpu().numpy()}")
            
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            # Debugging: Print input shapes if it fails
            print(f"Debug - MRI shape: {dummy_mri.shape}")
            print(f"Debug - Clinical shape: {dummy_clinical.shape}")

if __name__ == "__main__":
    main()