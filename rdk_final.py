#!/usr/bin/env python3
# rdk_final.py - COMPLETELY FIXED VERSION
import Hobot.GPIO as GPIO
import cv2
import time
import os
from datetime import datetime
import socket
import json
import subprocess

# CONFIGURE THIS!
COMPUTER_IP = '192.168.127.12'  # Your computer's IP
PORT = 5001

# GPIO Pins
CAPTURE_BUTTON = 13    # Button 1: Take photo
FILTER_BUTTON = 15     # Button 2: Cycle filters
CAPTURE_LED = 31       # LED 1: Blinks on capture
FILTER_LED_1 = 32      # LED 2: Filter indicator 1
FILTER_LED_2 = 33      # LED 3: Filter indicator 2

class FilterSystem:
    """Manages filter selection and LED indicators"""
    def __init__(self):
        # Available filters
        self.filters = ['normal', 'bw', 'vintage']
        self.filter_names = {
            'normal': 'Normal Color',
            'bw': 'Black & White',
            'vintage': 'Vintage Sepia'
        }
        self.current_filter_index = 0
        
        # Track LED states in software
        self.led1_state = False
        self.led2_state = False
        
        # Filter change cooldown
        self.last_filter_change = 0
        self.filter_cooldown = 0.5  # seconds
    
    def get_current_filter(self):
        return self.filters[self.current_filter_index]
    
    def get_filter_display_name(self):
        return self.filter_names[self.get_current_filter()]
    
    def next_filter(self):
        """Cycle to next filter with cooldown"""
        current_time = time.time()
        if current_time - self.last_filter_change < self.filter_cooldown:
            return self.get_current_filter()
        
        self.current_filter_index = (self.current_filter_index + 1) % len(self.filters)
        self.last_filter_change = current_time
        return self.get_current_filter()
    
    def get_led_pattern(self):
        """Get LED pattern for current filter"""
        if self.get_current_filter() == 'normal':
            return [False, False]
        elif self.get_current_filter() == 'bw':
            return [True, False]
        elif self.get_current_filter() == 'vintage':
            return [False, True]
        return [False, False]
    
    def update_led_states(self):
        """Update tracked LED states"""
        pattern = self.get_led_pattern()
        self.led1_state = pattern[0]
        self.led2_state = pattern[1]
        return pattern
    
    def get_led_display_states(self):
        return ("ON" if self.led1_state else "OFF", 
                "ON" if self.led2_state else "OFF")

def setup_gpio():
    """Initialize all GPIO pins"""
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    
    # Setup buttons as INPUT
    GPIO.setup(CAPTURE_BUTTON, GPIO.IN)
    GPIO.setup(FILTER_BUTTON, GPIO.IN)
    
    # Setup LEDs as OUTPUT
    GPIO.setup(CAPTURE_LED, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(FILTER_LED_1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(FILTER_LED_2, GPIO.OUT, initial=GPIO.LOW)
    
    print("✓ GPIO initialized")
    return True

def clear_camera_buffer(cap):
    """Clear camera buffer to get fresh frame - FIX FOR DUPLICATION"""
    if cap and cap.isOpened():
        # Grab several frames to clear buffer
        for _ in range(5):
            cap.grab()
        return True
    return False

def capture_fresh_photo(cap, filter_system):
    """Capture a FRESH photo - FIXED VERSION"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filter_name = filter_system.get_current_filter()
    filename = f"photo_{timestamp}_{filter_name}.jpg"
    
    print(f"\n📸 CAPTURING FRESH PHOTO...")
    print(f"   Filter: {filter_system.get_filter_display_name()}")
    
    # Clear camera buffer FIRST
    if not clear_camera_buffer(cap):
        print("  ⚠ Could not clear camera buffer")
    
    # Take fresh photo
    photo_captured = False
    for attempt in range(3):  # Try up to 3 times
        if cap and cap.isOpened():
            # Use grab() then retrieve() for more reliable capture
            if cap.grab():
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    cv2.imwrite(filename, frame)
                    if os.path.exists(filename):
                        file_size = os.path.getsize(filename)
                        print(f"  ✅ Fresh capture attempt {attempt+1}: {file_size} bytes")
                        photo_captured = True
                        break
            time.sleep(0.1)
    
    # Fallback to fswebcam
    if not photo_captured:
        print("  ⚠ OpenCV failed, using fswebcam...")
        subprocess.run(['fswebcam', '-r', '640x480', '--no-banner', '--skip', '3', filename], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return filename if os.path.exists(filename) else None

def blink_led(pin, times=3, duration=0.1):
    """Blink an LED"""
    for _ in range(times):
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(duration)

def update_hardware_leds(filter_system):
    """Update physical LEDs based on filter"""
    pattern = filter_system.update_led_states()
    GPIO.output(FILTER_LED_1, GPIO.HIGH if pattern[0] else GPIO.LOW)
    GPIO.output(FILTER_LED_2, GPIO.HIGH if pattern[1] else GPIO.LOW)

def flash_filter_leds():
    """Flash both filter LEDs to indicate change"""
    GPIO.output(FILTER_LED_1, GPIO.HIGH)
    GPIO.output(FILTER_LED_2, GPIO.HIGH)
    time.sleep(0.15)
    GPIO.output(FILTER_LED_1, GPIO.LOW)
    GPIO.output(FILTER_LED_2, GPIO.LOW)
    time.sleep(0.05)

def display_status(filter_system, capture_count, filter_count):
    """Display current system status"""
    filter_display = filter_system.get_filter_display_name()
    led1_disp, led2_disp = filter_system.get_led_display_states()
    
    print(f"Filter: {filter_display:15} | LEDs: {led1_disp}/{led2_disp} | Photos: {capture_count} | Filter changes: {filter_count}", end='\r')

def send_to_computer(filename, filter_system):
    """Send photo to computer"""
    if not os.path.exists(filename):
        print(f"  ❌ File not found: {filename}")
        return False
    
    try:
        print(f"  📤 Sending to {COMPUTER_IP}:{PORT}...")
        
        s = socket.socket()
        s.settimeout(5)
        s.connect((COMPUTER_IP, PORT))
        
        metadata = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'filter': filter_system.get_current_filter(),
            'filter_display': filter_system.get_filter_display_name()
        }
        
        s.send(json.dumps(metadata).encode() + b'\n')
        
        with open(filename, 'rb') as f:
            while chunk := f.read(4096):
                s.send(chunk)
        
        s.close()
        os.remove(filename)
        print(f"  ✅ Sent successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Send error: {e}")
        return False

class RDKCapture:
    def __init__(self, computer_ip='192.168.127.12', port=5001):
        self.computer_ip = computer_ip
        self.port = port
        self.filter_system = FilterSystem()
        self.cap = None
        
        print("=" * 60)
        print("RDK PHOTO CAPTURE - FIXED VERSION")
        print("=" * 60)
        print(f"Target: {computer_ip}:{port}")
        print("\nCONTROLS:")
        print("  Button 1 (Pin 13): Capture FRESH photo")
        print("  Button 2 (Pin 15): Change filter (any time)")
        print("\nFEATURES:")
        print("  • No photo duplication")
        print("  • Filter changes anytime")
        print("  • Clear camera buffer each capture")
        print("=" * 60)
    
    def initialize(self):
        """Initialize hardware"""
        if not setup_gpio():
            return False
        
        update_hardware_leds(self.filter_system)
        
        # Initialize camera with retry
        for attempt in range(3):
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Clear initial buffer
                clear_camera_buffer(self.cap)
                time.sleep(1)
                print("✓ Camera ready (buffer cleared)")
                return True
            time.sleep(0.5)
        
        print("⚠ Camera not found, using fswebcam fallback")
        return True
    
    def run(self):
        """Main event loop - COMPLETELY FIXED"""
        print("\n" + "=" * 40)
        print("SYSTEM READY - NO AUTO FILTER CYCLING")
        print("=" * 40)
        print("Controls work independently:")
        print("• Press Button 1 anytime to capture")
        print("• Press Button 2 anytime to change filter")
        print("=" * 40 + "\n")
        
        # Wait a moment to ensure buttons are stable
        time.sleep(0.5)
        
        # Initialize button states from CURRENT reading (not default)
        last_button1_state = GPIO.input(CAPTURE_BUTTON)
        last_button2_state = GPIO.input(FILTER_BUTTON)
        
        # Counters
        capture_count = 0
        filter_change_count = 0
        
        # Button cooldowns
        button1_cooldown = 0
        button2_cooldown = 0
        cooldown_time = 0.3  # seconds
        
        system_running = True
        last_status_update = 0
        
        try:
            while system_running:
                current_time = time.time()
                
                # Read button states
                current_button1 = GPIO.input(CAPTURE_BUTTON)
                current_button2 = GPIO.input(FILTER_BUTTON)
                
                # Update status display every 0.1 seconds
                if current_time - last_status_update > 0.1:
                    display_status(self.filter_system, capture_count, filter_change_count)
                    last_status_update = current_time
                
                # --- BUTTON 1: CAPTURE PHOTO ---
                # Detect rising edge (0→1) and check cooldown
                if (current_button1 == GPIO.HIGH and 
                    last_button1_state == GPIO.LOW and
                    current_time > button1_cooldown):
                    
                    button1_cooldown = current_time + cooldown_time
                    capture_count += 1
                    
                    print(f"\n\n" + "="*40)
                    print(f"🎯 CAPTURE #{capture_count}")
                    print("="*40)
                    
                    # Visual feedback
                    blink_led(CAPTURE_LED, 2, 0.08)
                    
                    # Capture FRESH photo
                    filename = capture_fresh_photo(self.cap, self.filter_system)
                    if filename:
                        # Send to computer
                        print("  📨 Sending to computer...")
                        if send_to_computer(filename, self.filter_system):
                            print(f"  ✅ Capture #{capture_count} complete")
                        else:
                            print(f"  ❌ Send failed, keeping local copy")
                    else:
                        print("  ❌ Capture failed")
                    
                    print("="*40 + "\n")
                
                # --- BUTTON 2: CHANGE FILTER ---
                # Detect rising edge (0→1) and check cooldown
                if (current_button2 == GPIO.HIGH and 
                    last_button2_state == GPIO.LOW and
                    current_time > button2_cooldown):
                    
                    button2_cooldown = current_time + cooldown_time
                    filter_change_count += 1
                    
                    print(f"\n" + "-"*30)
                    print(f"🔄 FILTER CHANGE #{filter_change_count}")
                    
                    # Change filter
                    old_filter = self.filter_system.get_filter_display_name()
                    self.filter_system.next_filter()
                    new_filter = self.filter_system.get_filter_display_name()
                    
                    print(f"  {old_filter} → {new_filter}")
                    
                    # Update LEDs with feedback
                    update_hardware_leds(self.filter_system)
                    flash_filter_leds()
                    update_hardware_leds(self.filter_system)
                    
                    print("-"*30 + "\n")
                
                # Update last states
                last_button1_state = current_button1
                last_button2_state = current_button2
                
                # Small delay
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n" + "="*50)
            print("SHUTDOWN REQUESTED")
            system_running = False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            system_running = False
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        print("\nCleaning up resources...")
        
        if self.cap:
            self.cap.release()
            print("✓ Camera released")
        
        # Turn off LEDs
        GPIO.output(CAPTURE_LED, GPIO.LOW)
        GPIO.output(FILTER_LED_1, GPIO.LOW)
        GPIO.output(FILTER_LED_2, GPIO.LOW)
        
        GPIO.cleanup()
        print("✓ GPIO cleaned up")
        
        print("\n" + "="*50)
        print("✅ SYSTEM STOPPED SUCCESSFULLY")
        print("="*50)

def main():
    """Main function"""
    print("\n" + "="*60)
    print("RDK ANIMAL PHOTO CAPTURE - FINAL FIXED VERSION")
    print("="*60)
    
    # Create and run system
    capture = RDKCapture(computer_ip=COMPUTER_IP, port=PORT)
    
    if capture.initialize():
        capture.run()
    else:
        print("❌ Initialization failed")

if __name__ == "__main__":
    main()
