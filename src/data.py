import os
import json
import urllib.request
import pandas as pd
import numpy as np
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def clean_description(desc):
    if isinstance(desc, (list, np.ndarray)):
        elements = []
        for d in desc:
            if d is not None and not (isinstance(d, float) and np.isnan(d)):
                elements.append(str(d).strip())
        return " ".join(elements)
    if pd.isna(desc) or desc is None:
        return ""
    return str(desc).strip()

def extract_image_url(images_val):
    """
    Extracts the primary image URL from metadata.
    Format could be a dictionary or a list.
    """
    if not images_val:
        return None
    if isinstance(images_val, dict):
        for key in ['large', 'medium', 'hi_res', 'thumb']:
            if key in images_val:
                val = images_val[key]
                if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                    url = val[0]
                    if isinstance(url, str) and url.strip():
                        return url
    elif isinstance(images_val, (list, np.ndarray)) and len(images_val) > 0:
        first_item = images_val[0]
        if isinstance(first_item, dict):
            for key in ['large', 'medium', 'hi_res', 'thumb']:
                if key in first_item:
                    val = first_item[key]
                    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                        url = val[0]
                        if isinstance(url, str) and url.strip():
                            return url
                    elif isinstance(val, str) and val.strip():
                        return val
        elif isinstance(first_item, str):
            return first_item
    return None

def clean_price(price_val):
    if pd.isna(price_val) or price_val is None:
        return np.nan
    if isinstance(price_val, (int, float)):
        return float(price_val)
    price_str = str(price_val).replace("$", "").replace(",", "").strip()
    try:
        return float(price_str)
    except ValueError:
        return np.nan

def load_and_preprocess_raw(max_products=15000, max_reviews=50000):
    print("Loading raw metadata (direct parquet read)...")
    # Read the first metadata chunk from Hugging Face (contains ~160k products, which is more than enough)
    url = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw_meta_Electronics/full-00000-of-00010.parquet"
    meta_df = pd.read_parquet(url)
    print(f"Loaded metadata for {len(meta_df)} products.")
    
    # Create a set of product IDs for O(1) membership checking
    meta_product_ids = set(meta_df['parent_asin'].unique())
    
    print("Loading and filtering raw reviews (streaming)...")
    reviews_ds = load_dataset(
        "json",
        data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Electronics.jsonl",
        split="train",
        streaming=True
    )
    
    reviews_list = []
    # Collect only reviews whose product matches our loaded metadata
    for item in tqdm(reviews_ds, desc="Filtering reviews"):
        p_asin = item.get('parent_asin')
        if p_asin in meta_product_ids:
            reviews_list.append(item)
            if len(reviews_list) >= max_reviews:
                break
                
    reviews_df = pd.DataFrame(reviews_list)
    print(f"Loaded {len(reviews_df)} matched reviews.")
    
    # Rename fields for compatibility with notebooks
    reviews_df = reviews_df.rename(columns={
        'rating': 'rating_rev', 
        'text': 'text_rev', 
        'helpful_vote': 'helpful_votes'
    })
    
    # Preprocess metadata columns
    meta_df['price_clean'] = meta_df['price'].apply(clean_price)
    meta_df['description_clean'] = meta_df['description'].apply(clean_description)
    meta_df['image_url'] = meta_df['images'].apply(extract_image_url)
    
    # Clean lists/dicts to strings for easy parquet storage
    meta_df['categories_str'] = meta_df['categories'].apply(lambda x: ", ".join(x) if isinstance(x, (list, np.ndarray)) else "")
    
    # Merge datasets
    merged_df = pd.merge(reviews_df, meta_df, on='parent_asin', how='inner', suffixes=('_rev', '_meta'))
    print(f"Merged dataset has {len(merged_df)} rows.")
    
    # Deduplicate and downsample to target limits
    unique_products = merged_df['parent_asin'].unique()
    if len(unique_products) > max_products:
        np.random.seed(42)
        selected_products = np.random.choice(unique_products, size=max_products, replace=False)
        merged_df = merged_df[merged_df['parent_asin'].isin(selected_products)]
    
    if len(merged_df) > max_reviews:
        merged_df = merged_df.sample(n=max_reviews, random_state=42)
        
    print(f"Subsampled dataset: {merged_df['parent_asin'].nunique()} products, {len(merged_df)} reviews.")
    return merged_df

def split_and_save_data(df, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform clean product-level split (no leakage)
    unique_products = np.array(df['parent_asin'].unique())
    np.random.seed(42)
    np.random.shuffle(unique_products)
    
    n_total = len(unique_products)
    n_val = int(n_total * 0.15)
    n_test = int(n_total * 0.15)
    
    val_products = unique_products[:n_val]
    test_products = unique_products[n_val:n_val + n_test]
    train_products = unique_products[n_val + n_test:]
    
    splits = {
        "train": list(train_products),
        "val": list(val_products),
        "test": list(test_products)
    }
    
    with open(os.path.join(output_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=4)
        
    # Map split back to dataframe
    split_map = {}
    for p in train_products:
        split_map[p] = "train"
    for p in val_products:
        split_map[p] = "val"
    for p in test_products:
        split_map[p] = "test"
        
    df['split'] = df['parent_asin'].map(split_map)
    
    # Save processed dataframe
    df.to_parquet(os.path.join(output_dir, "reviews_split.parquet"), index=False)
    
    # Extract product catalog mapping (metadata level)
    products_df = df[['parent_asin', 'title_meta', 'price_clean', 'description_clean', 'image_url', 'categories_str', 'split']].drop_duplicates('parent_asin')
    products_df.to_parquet(os.path.join(output_dir, "products_split.parquet"), index=False)
    
    print(f"Saved dataset splits to {output_dir}")
    print(f"Train products: {len(train_products)}, Val: {len(val_products)}, Test: {len(test_products)}")
    return df

def download_single_image(parent_asin, url, save_dir):
    if not url:
        return False
    ext = ".jpg"
    if ".png" in url.lower():
        ext = ".png"
    filepath = os.path.join(save_dir, f"{parent_asin}{ext}")
    
    # Skip if already exists
    if os.path.exists(filepath):
        return True
        
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception:
        return False

def cache_images(df, cache_dir="data/images", max_workers=10):
    os.makedirs(cache_dir, exist_ok=True)
    
    # Extract unique products with valid image URLs
    products = df[['parent_asin', 'image_url']].drop_duplicates('parent_asin')
    products = products[products['image_url'].notna() & (products['image_url'] != "")]
    
    print(f"Starting downloads for {len(products)} product images...")
    
    success_count = 0
    failures = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_image, row['parent_asin'], row['image_url'], cache_dir): row['parent_asin']
            for _, row in products.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            if future.result():
                success_count += 1
            else:
                failures += 1
                
    print(f"Image Caching Complete: {success_count} successfully downloaded, {failures} failures.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and preprocess Amazon Reviews and Metadata.")
    parser.add_argument("--max-products", type=int, default=15000, help="Maximum unique products to keep.")
    parser.add_argument("--max-reviews", type=int, default=50000, help="Maximum reviews to keep.")
    parser.add_argument("--skip-images", action="store_true", help="Skip downloading product images.")
    args = parser.parse_args()
    
    # Run pipeline
    df = load_and_preprocess_raw(max_products=args.max_products, max_reviews=args.max_reviews)
    df = split_and_save_data(df)
    
    if not args.skip_images:
        cache_images(df)

