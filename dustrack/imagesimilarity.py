"""
Image similarity and ordering methods for ultrasound images.
Optimized for grayscale ultrasound images with common dimensions.
This module was created to order training data and compare label consistency across similar images.
"""

import imagehash
from PIL import Image as PILImage
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import cv2
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from datanavigator import Image, ImageSequence

ORDERING_METHODS = ['imagehash', 'deep', 'hybrid', 'ssim', 'kmeans']

class ImageSequence(ImageSequence):
    """Encapsulates a sequence of images for processing."""
    def __init__(self, image_paths, grayscale=True, *args, **kwargs):
        super().__init__(image_paths, grayscale=grayscale, *args, **kwargs)

    def order(self, method="imagehash"):
        assert method in ORDERING_METHODS, "Unknown method"
        if method == 'imagehash':
            return ImageSequence(order_by_imagehash(self))
        elif method == 'deep':
            return ImageSequence(order_by_deep_features(self))
        elif method == 'hybrid':
            return ImageSequence(order_by_hybrid(self))
        elif method == 'kmeans':
            return ImageSequence(order_by_kmeans(self))
        elif method == 'ssim':
            return ImageSequence(order_by_ssim(self))
        else:
            raise ValueError("Unknown method")
    

class UltrasoundEncoder(nn.Module):
    """Simple CNN encoder for ultrasound image features."""
    def __init__(self, feature_dim=128):
        super().__init__()
        # Use pretrained ResNet backbone
        self.backbone = resnet18(weights='IMAGENET1K_V1')
        # Modify for grayscale input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace classifier with feature extractor
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feature_dim)
        
    def forward(self, x):
        return self.backbone(x)


def order_images_comparison(image_data, methods=['imagehash', 'deep', 'hybrid', 'ssim', 'kmeans'], preload=True):
    """
    Compare different ordering methods for ultrasound images.
    
    Args:
        image_data (list): Either list of image paths, or list of Image instances
        methods (list): Methods to compare. Available: 'imagehash', 'hybrid', 'deep', 'kmeans', 'ssim'
        preload (bool): If True and image_data is paths, preload images once and use across all methods
    
    Returns:
        dict: Results for each method
    """
    # Handle input format
    if isinstance(image_data, list) and len(image_data) > 0 and isinstance(image_data[0], Image):
        # Already preloaded: list of Image instances
        data_to_use = image_data
    else:
        # List of paths
        image_paths = image_data
        if preload:
            print("Preloading images once for all methods...")
            data_to_use = ImageSequence(image_paths, grayscale=True) # use grayscale for ultrasound images
        else:
            data_to_use = image_paths
    
    results = {}
    
    # Process methods in the order they are listed
    for method in methods:
        if method == 'imagehash':
            print(f"\n=== Processing method: {method} ===")
            results['imagehash'] = order_by_imagehash(data_to_use)
        elif method == 'hybrid':
            print(f"\n=== Processing method: {method} ===")
            results['hybrid'] = order_by_hybrid(data_to_use)
        elif method == 'deep':
            print(f"\n=== Processing method: {method} ===")
            results['deep'] = order_by_deep_features(data_to_use)
        elif method == 'kmeans':
            print(f"\n=== Processing method: {method} ===")
            results['kmeans'] = order_by_kmeans(data_to_use)
        elif method == 'ssim':
            print(f"\n=== Processing method: {method} ===")
            results['ssim'] = order_by_ssim(data_to_use)
        else:
            print(f"Warning: Unknown method '{method}' - skipping")
    
    return results

def order_by_imagehash(image_data, hash_size=32):
    """Order using perceptual hashing - best for brightness invariance."""
    
    # Handle input format
    if isinstance(image_data, list) and len(image_data) > 0 and isinstance(image_data[0], Image):
        # List of Image instances
        images = image_data
        use_preloaded = True
    else:
        # List of paths
        image_paths = image_data
        use_preloaded = False
    
    def compute_hash(image_path):
        try:
            with PILImage.open(image_path) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Use difference hash - more robust for ultrasound textures
                return imagehash.dhash(img, hash_size=hash_size)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def compute_hash_from_array(img_array):
        try:
            # Convert numpy array to PIL Image
            img = PILImage.fromarray(img_array)
            return imagehash.dhash(img, hash_size=hash_size)
        except Exception as e:
            print(f"Error processing image array: {e}")
            return None
    
    # Compute hashes
    hashes = []
    final_images = []
    
    if use_preloaded:
        print("Computing image hashes from preloaded images...")
        for img in tqdm(images, desc="Processing images"):
            hash_val = compute_hash_from_array(img.asnumpy2d())
            if hash_val is not None:
                hashes.append(hash_val)
                final_images.append(img)
    else:
        print("Computing image hashes...")
        for path in tqdm(image_paths, desc="Processing images"):
            hash_val = compute_hash(path)
            if hash_val is not None:
                hashes.append(hash_val)
                final_images.append(path)
    
    if len(hashes) < 2:
        return final_images
    
    # Distance matrix
    n = len(hashes)
    distances = np.zeros((n, n))
    
    print("Computing distance matrix...")
    for i in tqdm(range(n), desc="Distance calculation"):
        for j in range(i+1, n):
            # Hamming distance between hashes
            dist = hashes[i] - hashes[j]
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Hierarchical clustering
    condensed_distances = squareform(distances)
    linkage_matrix = linkage(condensed_distances, method='average')
    order = leaves_list(linkage_matrix)
    
    return [final_images[i] for i in order]

def order_by_deep_features(image_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Order using deep CNN features - best for complex anatomical patterns."""
    
    # Handle input format
    if isinstance(image_data, list) and len(image_data) > 0 and isinstance(image_data[0], Image):
        # List of Image instances
        images = image_data
        use_preloaded = True
    else:
        # List of paths
        image_paths = image_data
        use_preloaded = False
    
    # Load pretrained model
    model = UltrasoundEncoder(feature_dim=256)
    model.eval()
    model.to(device)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # ImageNet grayscale stats
    ])
    
    def extract_features_from_path(image_path):
        try:
            img = PILImage.open(image_path)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model(img_tensor)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def extract_features_from_array(img_array):
        try:
            img = PILImage.fromarray(img_array)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model(img_tensor)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing image array: {e}")
            return None
    
    # Extract features
    features = []
    final_images = []
    
    if use_preloaded:
        print("Extracting deep features from preloaded images...")
        for img in tqdm(images, desc="Processing images"):
            feat = extract_features_from_array(img.asnumpy2d())
            if feat is not None:
                features.append(feat)
                final_images.append(img)
    else:
        print("Extracting deep features...")
        for path in tqdm(image_paths, desc="Processing images"):
            feat = extract_features_from_path(path)
            if feat is not None:
                features.append(feat)
                final_images.append(path)
    
    if len(features) < 2:
        return final_images
    
    # Compute cosine similarity (better for deep features)
    print("Computing similarity matrix...")
    features_array = np.array(features)
    similarity_matrix = cosine_similarity(features_array)
    
    # Convert to distance matrix
    distance_matrix = 1 - similarity_matrix
    
    # Ensure diagonal is exactly zero (fix floating point precision issues)
    np.fill_diagonal(distance_matrix, 0)
    
    # Hierarchical clustering
    print("Performing hierarchical clustering...")
    condensed_distances = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distances, method='ward')
    order = leaves_list(linkage_matrix)
    
    return [final_images[i] for i in order]

def order_by_hybrid(image_data, n_components=50):
    """
    Hybrid approach: K-means for rough grouping, then hierarchical within groups.
    Optimized for grayscale ultrasound images of same dimensions.
    
    Args:
        image_data (list): Either list of image paths or list of Image instances
        n_components (int): PCA components for dimensionality reduction
    
    Returns:
        list: Ordered images (Image instances or paths)
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    # Handle input format
    if isinstance(image_data, list) and len(image_data) > 0 and isinstance(image_data[0], Image):
        # List of Image instances
        images = image_data
        use_preloaded = True
    else:
        # List of paths
        image_paths = image_data
        use_preloaded = False
    
    def load_and_preprocess_from_path(image_path):
        """Load and preprocess ultrasound image from path."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Normalize intensity - good for ultrasound
        img = cv2.equalizeHist(img)
        
        # Optional: Focus on central region (common in ultrasound)
        h, w = img.shape
        center_crop = img[h//4:3*h//4, w//4:3*w//4]
        
        return center_crop.flatten()
    
    def preprocess_from_array(img_array):
        """Preprocess ultrasound image from array."""
        # Normalize intensity - good for ultrasound
        img = cv2.equalizeHist(img_array)
        
        # Optional: Focus on central region (common in ultrasound)
        h, w = img.shape
        center_crop = img[h//4:3*h//4, w//4:3*w//4]
        
        return center_crop.flatten()
    
    # Load all images
    features = []
    final_images = []
    
    if use_preloaded:
        print("Loading and preprocessing preloaded images...")
        for img in tqdm(images, desc="Processing images"):
            feat = preprocess_from_array(img.asnumpy2d())
            if feat is not None:
                features.append(feat)
                final_images.append(img)
    else:
        print("Loading and preprocessing images...")
        for path in tqdm(image_paths, desc="Processing images"):
            feat = load_and_preprocess_from_path(path)
            if feat is not None:
                features.append(feat)
                final_images.append(path)
    
    if len(features) < 2:
        return final_images
    
    features_array = np.array(features)
    
    # Dimensionality reduction (crucial for pixel-level similarity)
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=min(n_components, len(features), features_array.shape[1]), 
              svd_solver='randomized', random_state=42)  # Use randomized SVD for speed
    features_reduced = pca.fit_transform(features_array)
    
    # Use k-means for rough grouping, then hierarchical within groups
    n_clusters = min(int(np.sqrt(len(features)) / 2), 5)
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_reduced)
    
    ordered_images = []
    for cluster_idx in range(n_clusters):
        cluster_mask = cluster_labels == cluster_idx
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 1:
            cluster_features = features_reduced[cluster_mask]
            cluster_distances = pdist(cluster_features, metric='euclidean')
            cluster_linkage = linkage(cluster_distances, method='ward')
            cluster_order = leaves_list(cluster_linkage)
            ordered_images.extend([final_images[cluster_indices[i]] for i in cluster_order])
        else:
            ordered_images.extend([final_images[i] for i in cluster_indices])
    
    return ordered_images

def order_by_kmeans(image_data, n_clusters=None, n_components=50):
    """
    Order images using K-means clustering with TSP-like ordering.
    
    Args:
        image_data (list): Either list of image paths or list of Image instances
        n_clusters (int): Number of clusters (auto if None)
        n_components (int): PCA components for dimensionality reduction
    
    Returns:
        list: Ordered images (Image instances or paths)
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    # Handle input format
    if isinstance(image_data, list) and len(image_data) > 0 and isinstance(image_data[0], Image):
        # List of Image instances
        images = image_data
        use_preloaded = True
    else:
        # List of paths
        image_paths = image_data
        use_preloaded = False
    
    def load_and_preprocess_from_path(image_path):
        """Load and preprocess ultrasound image from path."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Normalize intensity
        img = cv2.equalizeHist(img)
        return img.flatten()
    
    def preprocess_from_array(img_array):
        """Preprocess ultrasound image from array."""
        # Normalize intensity
        img = cv2.equalizeHist(img_array)
        return img.flatten()
    
    # Load all images
    features = []
    final_images = []
    
    if use_preloaded:
        print("Loading and preprocessing preloaded images for K-means...")
        for img in tqdm(images, desc="Processing images"):
            feat = preprocess_from_array(img.asnumpy2d())
            if feat is not None:
                features.append(feat)
                final_images.append(img)
    else:
        print("Loading and preprocessing images for K-means...")
        for path in tqdm(image_paths, desc="Processing images"):
            feat = load_and_preprocess_from_path(path)
            if feat is not None:
                features.append(feat)
                final_images.append(path)
    
    if len(features) < 2:
        return final_images
    
    features_array = np.array(features)
    
    # Dimensionality reduction
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=min(n_components, len(features), features_array.shape[1]),
              svd_solver='randomized', random_state=42)  # Use randomized SVD for speed
    features_reduced = pca.fit_transform(features_array)
    
    # Determine number of clusters
    if n_clusters is None:
        n_clusters = min(int(np.sqrt(len(features))), 10)
    
    print(f"K-means clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_reduced)
    
    # Order clusters by centroid similarity
    centroids = kmeans.cluster_centers_
    if len(centroids) > 1:
        centroid_distances = pdist(centroids)
        centroid_linkage = linkage(centroid_distances, method='ward')
        cluster_order = leaves_list(centroid_linkage)
    else:
        cluster_order = [0]
    
    # Within each cluster, order by similarity to centroid
    ordered_images = []
    for cluster_idx in cluster_order:
        cluster_mask = cluster_labels == cluster_idx
        cluster_features = features_reduced[cluster_mask]
        cluster_images = [final_images[i] for i in range(len(final_images)) if cluster_mask[i]]
        
        if len(cluster_features) > 1:
            # Order within cluster by distance to centroid
            centroid = centroids[cluster_idx]
            distances_to_centroid = np.linalg.norm(cluster_features - centroid, axis=1)
            cluster_order_indices = np.argsort(distances_to_centroid)
            ordered_images.extend([cluster_images[i] for i in cluster_order_indices])
        else:
            ordered_images.extend(cluster_images)
    
    return ordered_images

def compute_ssim_gpu_batch(images, batch_size=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Compute SSIM similarity matrix using GPU acceleration and batching.
    
    Args:
        images (list): List of preprocessed images (normalized to [0,1])
        batch_size (int): Batch size for GPU processing
        device (str): Device to use ('cuda' or 'cpu')
    
    Returns:
        np.ndarray: SSIM similarity matrix
    """
    try:
        import torch.nn.functional as F
        
        n = len(images)
        similarity_matrix = np.zeros((n, n), dtype=np.float32)
        
        # Convert images to tensors
        print("Converting images to tensors...")
        image_tensors = []
        for img in tqdm(images, desc="Converting to tensors"):
            if len(img.shape) == 2:  # Grayscale
                # Convert to 4D tensor: [1, 1, height, width]
                tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
            else:  # Color (shouldn't happen with grayscale images)
                # Convert to 4D tensor: [1, channels, height, width] 
                tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
            image_tensors.append(tensor)
        
        def ssim_torch(img1, img2, window_size=11, size_average=True):
            """Compute SSIM using PyTorch (GPU accelerated)."""
            def gaussian_kernel(size, sigma=1.5):
                coords = torch.arange(size, dtype=torch.float32, device=img1.device)
                coords -= size // 2
                g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
                g /= g.sum()
                # Create 2D kernel for 2D convolution
                kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
                return kernel_2d.unsqueeze(0).unsqueeze(0)
            
            window = gaussian_kernel(window_size).to(img1.device)
            
            # Ensure images are 4D: [batch, channel, height, width]
            if len(img1.shape) == 3:
                img1 = img1.unsqueeze(0)
            if len(img2.shape) == 3:
                img2 = img2.unsqueeze(0)
            
            # Use padding and stride for 2D convolution
            padding = window_size // 2
            mu1 = F.conv2d(img1, window, padding=padding, stride=1)
            mu2 = F.conv2d(img2, window, padding=padding, stride=1)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, stride=1) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, stride=1) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=padding, stride=1) - mu1_mu2
            
            C1 = (0.01) ** 2
            C2 = (0.03) ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            if size_average:
                return ssim_map.mean()
            else:
                return ssim_map.mean(dim=[1, 2, 3])
        
        print("Computing SSIM matrix using GPU...")
        with torch.no_grad():
            for i in tqdm(range(n), desc="SSIM calculation"):
                # Process diagonal
                similarity_matrix[i, i] = 1.0
                
                # Process row in batches
                for j_start in range(i + 1, n, batch_size):
                    j_end = min(j_start + batch_size, n)
                    
                    # Compute SSIM for batch
                    img1 = image_tensors[i]
                    batch_similarities = []
                    
                    for j in range(j_start, j_end):
                        img2 = image_tensors[j]
                        ssim_val = ssim_torch(img1, img2).cpu().item()
                        batch_similarities.append(ssim_val)
                    
                    # Fill matrix (symmetric)
                    for idx, j in enumerate(range(j_start, j_end)):
                        ssim_val = batch_similarities[idx]
                        similarity_matrix[i, j] = ssim_val
                        similarity_matrix[j, i] = ssim_val
        
        return similarity_matrix
        
    except Exception as e:
        print(f"GPU SSIM failed: {e}. Falling back to CPU implementation.")
        return None

def compute_ssim_parallel(images, n_workers=None):
    """
    Compute SSIM similarity matrix using parallel CPU processing.
    
    Args:
        images (list): List of preprocessed images
        n_workers (int): Number of worker processes
    
    Returns:
        np.ndarray: SSIM similarity matrix
    """
    from skimage.metrics import structural_similarity as ssim
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Limit to 8 to avoid memory issues
    
    n = len(images)
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    def compute_ssim_pair(args):
        i, j, img1, img2 = args
        return i, j, ssim(img1, img2, data_range=1.0)
    
    # Generate all unique pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, images[i], images[j]))
    
    print(f"Computing SSIM matrix using {n_workers} CPU workers...")
    
    # Fill diagonal
    for i in range(n):
        similarity_matrix[i, i] = 1.0
    
    # Process pairs in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(compute_ssim_pair, pairs),
            total=len(pairs),
            desc="SSIM calculation"
        ))
    
    # Fill similarity matrix
    for i, j, ssim_val in results:
        similarity_matrix[i, j] = ssim_val
        similarity_matrix[j, i] = ssim_val
    
    return similarity_matrix

def order_by_ssim(image_data, resize_shape=(256, 256), use_gpu=True, batch_size=50):
    """
    Order images using Structural Similarity Index (SSIM).
    
    Args:
        image_data (list): Either list of image paths or list of Image instances
        resize_shape (tuple): Shape to resize images for consistent comparison
        use_gpu (bool): Use GPU acceleration if available
        batch_size (int): Batch size for GPU processing
    
    Returns:
        list: Ordered images (Image instances or paths)
    """
    # Handle input format
    if isinstance(image_data, list) and len(image_data) > 0 and isinstance(image_data[0], Image):
        # List of Image instances
        images = image_data
        use_preloaded = True
    else:
        # List of paths
        image_paths = image_data
        use_preloaded = False
    
    def load_and_preprocess_from_path(image_path):
        """Load and preprocess image for SSIM from path."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Resize for consistent comparison
        img = cv2.resize(img, resize_shape)
        
        # Normalize to [0, 1] range
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def preprocess_from_array(img_array):
        """Preprocess image for SSIM from array."""
        # Resize for consistent comparison
        img = cv2.resize(img_array, resize_shape)
        
        # Normalize to [0, 1] range
        img = img.astype(np.float32) / 255.0
        
        return img
    
    # Load all images
    processed_images = []
    final_images = []
    
    if use_preloaded:
        print("Loading preloaded images for SSIM comparison...")
        for img in tqdm(images, desc="Loading images"):
            processed_img = preprocess_from_array(img.asnumpy2d())
            if processed_img is not None:
                processed_images.append(processed_img)
                final_images.append(img)
    else:
        print("Loading images for SSIM comparison...")
        for path in tqdm(image_paths, desc="Loading images"):
            processed_img = load_and_preprocess_from_path(path)
            if processed_img is not None:
                processed_images.append(processed_img)
                final_images.append(path)
    
    if len(processed_images) < 2:
        return final_images
    
    # Compute SSIM similarity matrix
    if use_gpu and torch.cuda.is_available():
        print("Using GPU-accelerated SSIM computation...")
        similarity_matrix = compute_ssim_gpu_batch(processed_images, batch_size=batch_size)
        if similarity_matrix is None:
            # Fall back to parallel CPU if GPU fails
            print("Falling back to parallel CPU SSIM...")
            similarity_matrix = compute_ssim_parallel(processed_images)
    else:
        print("Using parallel CPU SSIM computation...")
        similarity_matrix = compute_ssim_parallel(processed_images)
    
    if similarity_matrix is None:
        print("SSIM computation failed. Returning original order.")
        return final_images
    
    # Convert similarity to distance matrix
    distance_matrix = 1 - similarity_matrix
    
    # Ensure diagonal is exactly zero
    np.fill_diagonal(distance_matrix, 0)
    
    # Hierarchical clustering
    print("Performing hierarchical clustering...")
    condensed_distances = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distances, method='average')
    order = leaves_list(linkage_matrix)
    
    return [final_images[i] for i in order]
