#!/usr/bin/env python3
"""
Computer Server for RDK Animal Photo Classifier - WITH FILTER SUPPORT
Run on your computer: python computer_server.py
"""

import socket
import json
import threading
import os
import time
from datetime import datetime
import cv2
import numpy as np

# Import your existing modules
from image_classifier import ImageClassifier
from image_filters import apply_filter

class ComputerServer:
    def __init__(self, host='0.0.0.0', port=5001):
        self.host = host
        self.port = port
        self.classifier = ImageClassifier()
        
        # Create directories
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('static/gallery', exist_ok=True)
        for folder in ['bird', 'dog', 'cat', 'other', 'test']:
            os.makedirs(f'static/gallery/{folder}', exist_ok=True)
        
        print("=" * 70)
        print("COMPUTER SERVER - AI Photo Processing WITH FILTERS")
        print("=" * 70)
        print(f"Listening on: {host}:{port}")
        print(f"Web Gallery: http://localhost:5000")
        print("\nWaiting for photos from RDK...")
        print("Supported filters: Normal, Black & White, Vintage")
        print("=" * 70)
    
    def handle_client(self, client_socket, client_address):
        """Handle incoming image from RDK"""
        print(f"\n📨 Connection from RDK: {client_address[0]}")
        
        try:
            # Receive metadata (first line ends with \n)
            metadata_data = b''
            while True:
                chunk = client_socket.recv(1)
                if chunk == b'\n':
                    break
                if not chunk:
                    break
                metadata_data += chunk
            
            if not metadata_data:
                print("  No metadata received")
                client_socket.close()
                return
            
            metadata = json.loads(metadata_data.decode())
            filename = metadata['filename']
            filter_type = metadata.get('filter', 'normal')
            filter_display = metadata.get('filter_display', 'Normal Color')
            
            print(f"  📸 Receiving: {filename}")
            print(f"  🎨 Filter: {filter_display}")
            
            # Receive image data
            image_data = b''
            client_socket.settimeout(2.0)
            
            while True:
                try:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    image_data += chunk
                except socket.timeout:
                    break
            
            # Save uploaded file
            upload_path = f"uploads/{filename}"
            with open(upload_path, 'wb') as f:
                f.write(image_data)
            
            file_size = len(image_data)
            print(f"  📦 Received: {file_size} bytes")
            
            if file_size > 1024:  # Basic validation
                # Process the image
                self.process_image(upload_path, filter_type, filter_display)
            else:
                print(f"  ❌ File too small: {file_size} bytes")
            
        except json.JSONDecodeError:
            print("  ❌ Invalid metadata")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        finally:
            client_socket.close()
    
    def process_image(self, image_path, filter_type, filter_display):
        """Process image: apply filter, classify with AI, organize"""
        print(f"  🔄 Processing image...")
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  ❌ Could not read image")
                return
            
            # Apply filter if needed
            if filter_type != 'normal':
                print(f"  🎨 Applying {filter_display} filter...")
                image = apply_filter(image, filter_type)
                # Save filtered version temporarily
                cv2.imwrite(image_path, image)
            
            # Classify with AI
            print(f"  🤖 Running AI classification...")
            category, label, confidence = self.classifier.classify_and_organize(image_path)
            
            print(f"  ✅ AI Result: {label}")
            print(f"  ✅ Category: {category.upper()} ({confidence:.2%} confidence)")
            print(f"  🎨 Filter: {filter_display}")
            print(f"  🌐 Gallery: http://localhost:5000/#{category}")
            print(f"  📁 File moved to: static/gallery/{category}/")
            
            # Save metadata for web display
            self.save_image_metadata(image_path, category, filter_display, label, confidence)
            
            # Clean up upload
            if os.path.exists(image_path):
                os.remove(image_path)
                
        except Exception as e:
            print(f"  ❌ Processing error: {e}")
    
    def save_image_metadata(self, image_path, category, filter_name, label, confidence):
        """Save additional metadata for web display"""
        try:
            # Extract filename
            filename = os.path.basename(image_path)
            
            # Create metadata file
            metadata = {
                'filename': filename,
                'category': category,
                'filter': filter_name,
                'ai_label': label,
                'confidence': f"{confidence:.2%}",
                'timestamp': datetime.now().isoformat()
            }
            
            # Save as JSON in same directory
            metadata_path = image_path.replace('.jpg', '.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"  Note: Could not save metadata: {e}")
    
    def run(self):
        """Start server to receive images from RDK"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        
        print(f"✅ Server started on port {self.port}")
        print(f"📡 Ready for RDK connections...")
        
        try:
            while True:
                client, addr = server.accept()
                thread = threading.Thread(target=self.handle_client, args=(client, addr))
                thread.daemon = True
                thread.start()
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down server...")
        finally:
            server.close()
            print("✅ Server stopped")

if __name__ == "__main__":
    server = ComputerServer()