import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mobile_sam import sam_model_registry, SamPredictor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
import warnings
import os
from tqdm import tqdm
import torch.nn.functional as F
import streamlit as st
import io

# Suppress warnings
warnings.filterwarnings("ignore")

class BarnacleDataset(Dataset):
    """Modified Dataset class to work with Streamlit UploadedFile objects"""
    
    def __init__(self, uploaded_images, uploaded_masks, transform=None):
        """
        Args:
            uploaded_images: List of Streamlit UploadedFile objects (images)
            uploaded_masks: List of Streamlit UploadedFile objects (masks)
        """
        # Sort by filename to ensure proper pairing
        self.uploaded_images = sorted(uploaded_images, key=lambda x: self._extract_number(x.name))
        self.uploaded_masks = sorted(uploaded_masks, key=lambda x: self._extract_number(x.name))
        self.transform = transform
        
        # Validate that we have matching pairs
        if len(self.uploaded_images) != len(self.uploaded_masks):
            raise ValueError(f"Number of images ({len(self.uploaded_images)}) doesn't match number of masks ({len(self.uploaded_masks)})")
    
    def _extract_number(self, filename):
        # Extract ordered number from filename for sorting
        import re
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else 0
        
    def __len__(self):
        return len(self.uploaded_images) # return amount of images
    
    def __getitem__(self, idx):
        # Load image from UploadedFile
        image_file = self.uploaded_images[idx]
        image_file.seek(0)  
        image = np.array(Image.open(image_file).convert('RGB')) 
        
        # Load mask from UploadedFile
        mask_file = self.uploaded_masks[idx]
        mask_file.seek(0)  # Reset file pointer
        mask_img = np.array(Image.open(mask_file))
        
        # Convert blue outlines to binary mask
        if len(mask_img.shape) == 3:
            if mask_img.shape[2] == 4:  # RGBA case
                blue_channel = mask_img[:, :, 2] 
                alpha_channel = mask_img[:, :, 3]
                binary_mask = ((blue_channel > 50) & (alpha_channel > 0)).astype(np.uint8) # turn to true in boolean
            else:  # regular RGB case
                blue_channel = mask_img[:, :, 2]
                red_channel = mask_img[:, :, 0]
                green_channel = mask_img[:, :, 1]
                binary_mask = ((blue_channel > 50) & (blue_channel > red_channel) & (blue_channel > green_channel)).astype(np.uint8)# turn to true in boolean
        else:
            binary_mask = (mask_img > 50).astype(np.uint8) # turn to true in boolean
        
        # Find contours and fill them
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(binary_mask)
        cv2.fillPoly(filled_mask, contours, 1)
        
        # Keep original dimensions
        h, w = image.shape[:2]
        
        # Resize to the input size that SAM expects
        target_size = 1024
        image_resized = cv2.resize(image, (target_size, target_size))
        mask_resized = cv2.resize(filled_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized).float()
        
        return image_tensor, mask_tensor, (h, w)

def dice_loss(pred, target, smooth=1.0):
    # Dice loss function to check binary mask overlap
    pred = torch.sigmoid(pred)
    
    # Flatten tensors
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    # focal loss function to avoid abundant background in the image
    pred = torch.sigmoid(pred)
    
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    ce_loss = F.binary_cross_entropy(pred_flat, target_flat, reduction='none')
    p_t = pred_flat * target_flat + (1 - pred_flat) * (1 - target_flat) # confidence calculation
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0: # apply weights
        alpha_t = alpha * target_flat + (1 - alpha) * (1 - target_flat)
        loss = alpha_t * loss
    
    return loss.mean()

class FineTunedMobileSAM(nn.Module):
    # model class
    
    def __init__(self, sam_model):
        super().__init__()
        self.sam = sam_model
        
        # Freeze most of SAM model, only fine-tune mask decoder
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
            
        # Only fine-tune mask decoder
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
    
    def preprocess_image(self, image_np): # necessary for model input
        # Resize to 1024 while maintaining aspect ratio and pad
        h, w = image_np.shape[:2]
        scale = 1024 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image_np, (new_w, new_h))
        
        # Pad to 1024x1024
        padded = np.zeros((1024, 1024, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        return padded, (new_h, new_w), (h, w)
    
    def forward(self, images, original_sizes): # forward pass of the model
        batch_size = images.shape[0]
        all_masks = []
        
        # Process each image in the batch
        for i in range(batch_size):
            # Convert tensor back to numpy for preprocessing
            image_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Preprocess image
            preprocessed_img, processed_size, orig_size = self.preprocess_image(image_np)
            
            # Convert back to tensor
            img_tensor = torch.from_numpy(preprocessed_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(images.device)
            
            # Get image embeddings
            with torch.no_grad():
                image_embeddings = self.sam.image_encoder(img_tensor)
            
            # Create empty prompt embeddings (no points/boxes)
            sparse_embeddings = torch.zeros((1, 0, 256), device=images.device)
            dense_embeddings = torch.zeros((1, 256, 64, 64), device=images.device)
            
            # Generate masks using the mask decoder
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            # Resize mask to 1024x1024 first
            masks_1024 = F.interpolate(low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False)
            
            # Crop to processed size
            masks_cropped = masks_1024[0, 0, :processed_size[0], :processed_size[1]]
            
            # Resize to original size
            final_mask = F.interpolate(
                masks_cropped.unsqueeze(0).unsqueeze(0), 
                size=orig_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            
            all_masks.append(final_mask)
        
        # Stack all masks and add batch dimension
        final_masks = torch.stack(all_masks, dim=0).unsqueeze(1)
        return final_masks, torch.ones(batch_size, 1, device=images.device)

def train_model_streamlit(model, train_loader, num_epochs=50, learning_rate=1e-4, progress_container=None):
    # training for integration with streamlit
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.train()
    
    # Streamlit progress bars
    if progress_container:
        epoch_progress = progress_container.progress(0)
        status_text = progress_container.empty()
        loss_chart = progress_container.empty()
        
        # loss tracking
        epoch_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_losses = []
        
        for batch_idx, (images, masks, original_sizes) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_masks, iou_pred = model(images, original_sizes)
            
            # Ensure masks have the same dimensions
            if pred_masks.dim() != masks.dim():
                if masks.dim() == 2:
                    masks = masks.unsqueeze(0).unsqueeze(0)
                elif masks.dim() == 3:
                    masks = masks.unsqueeze(1)
            
            # Resize predictions to match target size if needed
            if pred_masks.shape[-2:] != masks.shape[-2:]:
                pred_masks = F.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            # Calculate losses
            dice_loss_val = dice_loss(pred_masks, masks)
            focal_loss_val = focal_loss(pred_masks, masks)
            total_loss_val = dice_loss_val + focal_loss_val
            
            # Backward pass
            total_loss_val.backward()
            optimizer.step()
            
            total_loss += total_loss_val.item()
            batch_losses.append(total_loss_val.item())
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        
        # Update Streamlit progress
        if progress_container:
            progress = (epoch + 1) / num_epochs
            epoch_progress.progress(progress)
            status_text.text(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}')
            
            # Update loss line chart 
            if len(epoch_losses) > 1:
                import pandas as pd
                loss_df = pd.DataFrame({'Epoch': range(1, len(epoch_losses) + 1), 'Loss': epoch_losses})
                loss_chart.line_chart(loss_df.set_index('Epoch'))
    
    return model

def predict_barnacles_from_uploaded(model, uploaded_file, device): # Streamlit functions
    
    # Load image from uploaded file
    uploaded_file.seek(0)
    image = np.array(Image.open(uploaded_file).convert('RGB'))
    original_size = image.shape[:2]
    
    # Resize to 1024x1024 to match training
    image_resized = cv2.resize(image, (1024, 1024))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        pred_masks, _ = model(image_tensor, [original_size])
        
        # Resize prediction back to original size
        pred_masks = F.interpolate(pred_masks, size=original_size, mode='bilinear', align_corners=False)
        pred_masks = torch.sigmoid(pred_masks)
        
        # Convert to binary mask
        binary_mask = (pred_masks.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    
    return binary_mask, image

def train_barnacle_model_streamlit(uploaded_images, uploaded_masks, num_epochs=50, learning_rate=1e-4, progress_container=None): # Train model function used in other script
    
    try:
        # Validate inputs
        if not uploaded_images or not uploaded_masks:
            raise ValueError("No images or masks provided")
        
        if len(uploaded_images) != len(uploaded_masks):
            raise ValueError(f"Number of images ({len(uploaded_images)}) doesn't match number of masks ({len(uploaded_masks)})")
        
        # Initialize MobileSAM
        if progress_container:
            progress_container.info("Loading MobileSAM model...")
        
        # Check if mobile_sam.pt exists, if not provide guidance
        if not os.path.exists("mobile_sam.pt"):
            raise FileNotFoundError("mobile_sam.pt not found. Please download it from the MobileSAM repository.")
        
        sam_model = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if progress_container:
            progress_container.success(f"Using device: {device}")
        
        # Create fine-tuned model
        model = FineTunedMobileSAM(sam_model)
        
        # Create dataset and dataloader
        dataset = BarnacleDataset(uploaded_images, uploaded_masks)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        if progress_container:
            progress_container.info(f"Created dataset with {len(dataset)} image-mask pairs")
        
        # Train the model
        if progress_container:
            progress_container.info("Starting training...")
        
        trained_model = train_model_streamlit(
            model, 
            train_loader, 
            num_epochs=num_epochs, 
            learning_rate=learning_rate,
            progress_container=progress_container
        )
        
        if progress_container:
            progress_container.success("Training completed!")
        
        return trained_model, device
        
    except Exception as e:
        if progress_container:
            progress_container.error(f"Training failed: {str(e)}")
        raise e

def save_model_streamlit(model, filename="barnacle_mobilesam.pth"):
    # Save trained model
    torch.save(model.state_dict(), filename)
    return filename

def load_model_streamlit(model_file, device):
    # Load a pretrained model
    if not os.path.exists("mobile_sam.pt"):
        raise FileNotFoundError("mobile_sam.pt not found. Please download it from the MobileSAM repository.")
    
    # Initialize base SAM model
    sam_model = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
    model = FineTunedMobileSAM(sam_model)
    
    # Load trained weights
    if hasattr(model_file, 'read'):  # UploadedFile object
        model_file.seek(0)
        state_dict = torch.load(io.BytesIO(model_file.read()), map_location=device)
    else:  # File path
        state_dict = torch.load(model_file, map_location=device)
    
    model.load_state_dict(state_dict)
    model.to(device)
    return model