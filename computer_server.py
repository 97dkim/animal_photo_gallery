#!/usr/bin/env python3
"""
Computer Server - DEBUG VERSION
"""

import socket
import json
import threading
import os
import time
from datetime import datetime
import cv2
import numpy as np

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
        for folder in ['bird', 'dog', 'cat', 'other_animal', 'non_animal']:
            os.makedirs(f'static/gallery/{folder}', exist_ok=True)
        
        print("=" * 70)
        print("COMPUTER SERVER - DEBUG MODE")
        print("=" * 70)
        print(f"Listening on: {host}:{port}")
        print(f"Web Gallery: http://localhost:5000")
        print(f"Start Time: {datetime.now().strftime('%H:%M:%S')}")
        print("\nServer is ACTIVE and waiting...")
        print("Press Ctrl+C to stop")
        print("=" * 70)
    
    def handle_client(self, client_socket, client_address):
        """Handle incoming image from RDK"""
        print(f"\n📨 NEW CONNECTION from {client_address[0]}")
        
        try:
            # Receive metadata
            print("  Receiving metadata...")
            metadata_data = b''
            
            while True:
                chunk = client_socket.recv(1)
                if chunk == b'\n':
                    break
                metadata_data += chunk
            
            if not metadata_data:
                print("  ❌ Empty metadata")
                return
            
            metadata = json.loads(metadata_data.decode())
            filename = metadata['filename']
            filter_type = metadata.get('filter', 'normal')
            filter_display = metadata.get('filter_display', 'Normal Color')
            
            print(f"  📸 Photo: {filename}")
            print(f"  🎨 Filter: {filter_display}")
            
            # Receive image data
            print("  Receiving image data...")
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
            
            # Save file
            upload_path = f"uploads/{filename}"
            with open(upload_path, 'wb') as f:
                f.write(image_data)
            
            file_size = len(image_data)
            print(f"  ✅ Received: {file_size} bytes")
            
            if file_size > 1024:
                self.process_image(upload_path, filter_type, filter_display)
            else:
                print(f"  ❌ File too small")
            
        except json.JSONDecodeError:
            print("  ❌ Invalid JSON metadata")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        finally:
            client_socket.close()
            print(f"  Connection closed")
    
    def process_image(self, image_path, filter_type, filter_display):
        """Process the received image"""
        print(f"  🔄 Processing image...")
        
        try:
            # Read and process
            image = cv2.imread(image_path)
            if image is None:
                print(f"  ❌ Could not read image")
                return
            
            # Apply filter
            if filter_type != 'normal':
                print(f"  🎨 Applying {filter_display} filter...")
                image = apply_filter(image, filter_type)
                cv2.imwrite(image_path, image)
            
            # AI classification
            print(f"  🤖 Running AI animal classification...")
            category, label, confidence = self.classifier.classify_and_organize(image_path)
            
            print(f"  ✅ AI Result: {label}")
            print(f"  ✅ Category: {category.upper()} ({confidence:.2%})")
            print(f"  🌐 Gallery: http://localhost:5000/#{category}")
            
            # Cleanup
            if os.path.exists(image_path):
                os.remove(image_path)
                
        except Exception as e:
            print(f"  ❌ Processing error: {e}")
    
    def run(self):
        """Start the server"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        
        print(f"\n✅ Server started successfully!")
        print(f"📡 Waiting for RDK connections on port {self.port}...")
        
        try:
            while True:
                # This will "block" until a connection arrives
                client, addr = server.accept()
                thread = threading.Thread(target=self.handle_client, args=(client, addr))
                thread.daemon = True
                thread.start()
                
        except KeyboardInterrupt:
            print("\n\n🛑 Server shutdown requested...")
        except Exception as e:
            print(f"\n❌ Server error: {e}")
        finally:
            server.close()
            print("✅ Server stopped")

if __name__ == "__main__":
    server = ComputerServer()
    server.run()