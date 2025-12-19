#!/usr/bin/env python3
"""
Web Gallery for RDK Animal Photo Classifier - WITH FILTER DISPLAY
Run on your computer: python web_gallery.py
"""

from flask import Flask, render_template, send_from_directory, jsonify
import os
import json
import time
from collections import OrderedDict

app = Flask(__name__)

# Define category order
CATEGORY_ORDER = ['dog', 'cat', 'bird', 'other_animal', 'non_animal']


def get_latest_mtime(gallery_path='static/gallery'):
    """Return latest modification time (float) across all gallery images."""
    latest = 0.0
    if not os.path.exists(gallery_path):
        return latest
    for root, _, files in os.walk(gallery_path):
        for fname in files:
            if fname.lower().endswith('.jpg'):
                fpath = os.path.join(root, fname)
                try:
                    latest = max(latest, os.path.getmtime(fpath))
                except OSError:
                    pass
    return latest

def load_image_metadata(image_path):
    """Load metadata for an image if it exists"""
    metadata_path = image_path.replace('.jpg', '.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

@app.route('/')
def gallery():
    """Main gallery page"""
    gallery_path = 'static/gallery'
    categories = OrderedDict()
    filter_stats = {}
    latest_mtime = 0.0
    
    # Get all categories in defined order
    all_dirs = os.listdir(gallery_path) if os.path.exists(gallery_path) else []
    sorted_categories = sorted(all_dirs, key=lambda x: CATEGORY_ORDER.index(x) if x in CATEGORY_ORDER else len(CATEGORY_ORDER))
    
    for category in sorted_categories:
        category_path = os.path.join(gallery_path, category)
        if os.path.isdir(category_path):
            images = []
            for img in sorted(os.listdir(category_path), reverse=True):  # Newest first
                if img.lower().endswith('.jpg'):
                    img_path = os.path.join(category_path, img)
                    img_url = f'/gallery/{category}/{img}'
                    img_time = os.path.getmtime(img_path)
                    latest_mtime = max(latest_mtime, img_time)
                    
                    # Load metadata
                    metadata = load_image_metadata(img_path) or {}
                    
                    images.append({
                        'url': img_url,
                        'name': img,
                        'category': category,
                        'time': time.strftime('%B %d, %Y. %I:%M%p', time.localtime(img_time)).replace(' 0', ' '),
                        'filter': metadata.get('filter', 'Normal'),
                        'ai_label': metadata.get('ai_label', 'Not classified'),
                        'confidence': metadata.get('confidence', 'N/A')
                    })
                    
                    # Track filter usage
                    filter_name = metadata.get('filter', 'Normal')
                    filter_stats[filter_name] = filter_stats.get(filter_name, 0) + 1
            
            if images:
                categories[category] = images
    
    if latest_mtime == 0.0:
        latest_mtime = get_latest_mtime(gallery_path)

    return render_template('gallery.html', 
                         categories=categories, 
                         filter_stats=filter_stats,
                         latest_mtime=latest_mtime)

@app.route('/gallery/<category>/<filename>')
def serve_image(category, filename):
    """Serve individual images"""
    return send_from_directory(f'static/gallery/{category}', filename)

@app.route('/download/<category>/<filename>')
def download_image(category, filename):
    """Serve image as attachment for download"""
    return send_from_directory(f'static/gallery/{category}', filename, as_attachment=True)

@app.route('/api/stats')
def get_stats():
    """Get gallery statistics"""
    gallery_path = 'static/gallery'
    stats = OrderedDict()
    total = 0
    filter_stats = {}
    
    all_dirs = os.listdir(gallery_path) if os.path.exists(gallery_path) else []
    sorted_categories = sorted(all_dirs, key=lambda x: CATEGORY_ORDER.index(x) if x in CATEGORY_ORDER else len(CATEGORY_ORDER))
    
    for category in sorted_categories:
        category_path = os.path.join(gallery_path, category)
        if os.path.isdir(category_path):
            count = 0
            for img in os.listdir(category_path):
                if img.lower().endswith('.jpg'):
                    count += 1
                    # Check for filter metadata
                    img_path = os.path.join(category_path, img)
                    metadata = load_image_metadata(img_path)
                    if metadata and 'filter' in metadata:
                        filter_name = metadata['filter']
                        filter_stats[filter_name] = filter_stats.get(filter_name, 0) + 1
            
            stats[category] = count
            total += count
    
    return jsonify({
        'categories': stats,
        'total': total,
        'filters': filter_stats
    })


@app.route('/api/last-updated')
def last_updated():
    """Return latest image modification time for conditional refresh."""
    latest = get_latest_mtime('static/gallery')
    return jsonify({'latest_mtime': latest})

@app.route('/category/<category_name>')
def category_view(category_name):
    """View images in a specific category"""
    category_path = f'static/gallery/{category_name}'
    if not os.path.exists(category_path):
        return "Category not found", 404
    
    images = []
    for img in sorted(os.listdir(category_path), reverse=True):
        if img.lower().endswith('.jpg'):
            img_path = os.path.join(category_path, img)
            img_url = f'/gallery/{category_name}/{img}'
            img_time = os.path.getmtime(img_path)
            
            # Load metadata
            metadata = load_image_metadata(img_path) or {}
            
            images.append({
                'url': img_url,
                'name': img,
                'filter': metadata.get('filter', 'Normal'),
                'ai_label': metadata.get('ai_label', 'Not classified'),
                'confidence': metadata.get('confidence', 'N/A'),
                'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(img_time))
            })
    
    return render_template('category.html', 
                         category=category_name, 
                         images=images)

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('static/gallery', exist_ok=True)
    for folder in ['bird', 'dog', 'cat', 'other', 'test']:
        os.makedirs(f'static/gallery/{folder}', exist_ok=True)
    
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 70)
    print("WEB GALLERY SERVER - WITH FILTER DISPLAY")
    print("=" * 70)
    print("Access points:")
    print("  http://localhost:5000       - Main gallery")
    print("  http://localhost:5000/api/stats - Statistics")
    print("  http://localhost:5000/category/<name> - Category view")
    print("\nFeatures:")
    print("  • Shows applied filter for each photo")
    print("  • Displays AI classification results")
    print("  • Filter usage statistics")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)