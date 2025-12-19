#!/usr/bin/env python3
"""
Main controller for RDK Animal Photo Classifier
Run this first: sudo python3 main.py
"""

import Hobot.GPIO as GPIO
import cv2
import time
import os
import threading
from datetime import datetime
from image_classifier import ImageClassifier
from image_filters import apply_filter

class PhotoCaptureSystem:
    def __init__(self):
        # Pin Definitions (BOARD numbering)
        self.CAPTURE_BUTTON = 23   # Button 1: Take photo
        self.FILTER_BUTTON = 24    # Button 2: Cycle filter
        self.CAPTURE_LED = 6     # LED 1: Blinks on capture
        self.FILTER_LED_1 = 27     # LED 2: Filter indicator 1
        self.FILTER_LED_2 = 22     # LED 3: Filter indicator 2
        
        # System state
        self.current_filter = 'normal'
        self.filters = ['normal', 'bw', 'vintage']
        self.filter_index = 0
        self.last_capture_time = 0
        self.capture_debounce = 1.0  # 1 second between captures
        
        # Create necessary directories
        os.makedirs('captures', exist_ok=True)
        os.makedirs('static/gallery', exist_ok=True)
        
        # Initialize components
        print("Initializing AI Classifier...")
        self.classifier = ImageClassifier()
        
        print("Initializing Camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Cannot open camera!")
            exit(1)
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(2)  # Camera warm-up
        
        # GPIO Setup
        print("Initializing GPIO...")
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        
        # Setup pins
        GPIO.setup(self.CAPTURE_BUTTON, GPIO.IN)
        GPIO.setup(self.FILTER_BUTTON, GPIO.IN)
        GPIO.setup(self.CAPTURE_LED, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.FILTER_LED_1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.FILTER_LED_2, GPIO.OUT, initial=GPIO.LOW)
        
        # Initial filter LED state
        self.update_filter_leds()
        
        print("\n=== RDK Animal Photo Classifier Ready ===")
        print(f"Current filter: {self.current_filter}")
        print("Press Button 1 (Pin 13) to capture photo")
        print("Press Button 2 (Pin 15) to change filter")
        print("=" * 40)
    
    def update_filter_leds(self):
        """Update LED indicators based on current filter"""
        if self.current_filter == 'normal':
            GPIO.output(self.FILTER_LED_1, GPIO.LOW)
            GPIO.output(self.FILTER_LED_2, GPIO.LOW)
        elif self.current_filter == 'bw':
            GPIO.output(self.FILTER_LED_1, GPIO.HIGH)
            GPIO.output(self.FILTER_LED_2, GPIO.LOW)
        elif self.current_filter == 'vintage':
            GPIO.output(self.FILTER_LED_1, GPIO.LOW)
            GPIO.output(self.FILTER_LED_2, GPIO.HIGH)
    
    def blink_led(self, pin, times=3, duration=0.15):
        """Blink an LED for visual feedback"""
        for _ in range(times):
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(duration)
    
    def cycle_filter(self):
        """Cycle to next filter and update indicators"""
        self.filter_index = (self.filter_index + 1) % len(self.filters)
        self.current_filter = self.filters[self.filter_index]
        print(f"Filter changed to: {self.current_filter}")
        self.update_filter_leds()
        
        # Quick blink of both filter LEDs to confirm change
        GPIO.output(self.FILTER_LED_1, GPIO.HIGH)
        GPIO.output(self.FILTER_LED_2, GPIO.HIGH)
        time.sleep(0.3)
        self.update_filter_leds()
    
    def capture_photo(self):
        """Capture, classify, and organize a photo"""
        current_time = time.time()
        if current_time - self.last_capture_time < self.capture_debounce:
            return
        
        self.last_capture_time = current_time
        print("\n" + "=" * 40)
        print("CAPTURING PHOTO...")
        
        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            return
        
        # Apply selected filter
        if self.current_filter != 'normal':
            print(f"Applying {self.current_filter} filter...")
            frame = apply_filter(frame, self.current_filter)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captures/photo_{timestamp}.jpg"
        
        # Save temporary image
        cv2.imwrite(filename, frame)
        print(f"Image saved temporarily: {filename}")
        
        # Blink capture LED to confirm
        self.blink_led(self.CAPTURE_LED, times=2, duration=0.1)
        
        # Classify and organize image using AI
        print("Running AI classification...")
        folder, label, confidence = self.classifier.classify_and_organize(filename)
        
        # Print results
        print("=" * 40)
        print(f"AI CLASSIFICATION RESULT:")
        print(f"  Detected: {label}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Category: {folder.upper()}")
        print(f"  Gallery: http://rdk.local/{folder}/")
        print("=" * 40)
        
        return True
    
    def run(self):
        """Main event loop"""
        last_filter_state = GPIO.LOW
        
        try:
            while True:
                # Check capture button (active HIGH when pressed)
                if GPIO.input(self.CAPTURE_BUTTON) == GPIO.HIGH:
                    self.capture_photo()
                    time.sleep(0.3)  # Simple debounce
                
                # Check filter button with edge detection
                current_filter_state = GPIO.input(self.FILTER_BUTTON)
                if current_filter_state == GPIO.HIGH and last_filter_state == GPIO.LOW:
                    self.cycle_filter()
                    time.sleep(0.3)  # Simple debounce
                
                last_filter_state = current_filter_state
                time.sleep(0.02)  # Small delay to reduce CPU usage
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        if self.cap:
            self.cap.release()
        GPIO.output(self.CAPTURE_LED, GPIO.LOW)
        GPIO.output(self.FILTER_LED_1, GPIO.LOW)
        GPIO.output(self.FILTER_LED_2, GPIO.LOW)
        GPIO.cleanup()
        print("System shutdown complete.")

if __name__ == "__main__":
    system = PhotoCaptureSystem()
    system.run()