#!/usr/bin/env python3
"""
XR Tank — PCA9685 Servo Controller (Pi 5 compatible)
Replaces the compiled _XiaoRGEEK_SERVO_.so binary

Hardware:
  U2: PCA9685 I2C PWM Generator @ 0x40
      8 channels (PWM0-PWM7) driving servos
  U4: AT24C02 I2C EEPROM @ 0x50
      Stores default servo angle positions

The original .so bit-banged I2C through RPi.GPIO.
This module uses smbus2 for proper I2C on Pi 5.

API (drop-in compatible with original):
    from servo_controller import XR_Servo
    servo = XR_Servo()
    servo.XiaoRGEEK_SetServoAngle(1, 90)   # Servo 1 to 90 degrees
    servo.XiaoRGEEK_SaveServo()              # Save all positions to EEPROM
    servo.XiaoRGEEK_ReSetServo()             # Restore saved positions
"""

import time
import struct
import logging
from typing import Optional, List

log = logging.getLogger("tank.servo")

# ---------------------------------------------------------------------------
# Conditional hardware import
# ---------------------------------------------------------------------------
try:
    import smbus2
    I2C_AVAILABLE = True
except ImportError:
    smbus2 = None
    I2C_AVAILABLE = False
    log.warning("smbus2 not available — servo controller in SIMULATION mode")

# ---------------------------------------------------------------------------
# PCA9685 Register Map
# ---------------------------------------------------------------------------
class PCA9685_REG:
    MODE1       = 0x00
    MODE2       = 0x01
    SUBADR1     = 0x02
    SUBADR2     = 0x03
    SUBADR3     = 0x04
    ALLCALLADR  = 0x05
    # Per-channel registers: base + 4*channel
    LED0_ON_L   = 0x06
    LED0_ON_H   = 0x07
    LED0_OFF_L  = 0x08
    LED0_OFF_H  = 0x09
    # ... channels 1-15 follow at +4 each
    ALL_LED_ON_L  = 0xFA
    ALL_LED_ON_H  = 0xFB
    ALL_LED_OFF_L = 0xFC
    ALL_LED_OFF_H = 0xFD
    PRESCALE      = 0xFE

    # MODE1 bits
    RESTART = 0x80
    SLEEP   = 0x10
    ALLCALL = 0x01
    AI      = 0x20  # Auto-increment

# ---------------------------------------------------------------------------
# PCA9685 Driver
# ---------------------------------------------------------------------------
class PCA9685:
    """
    Low-level PCA9685 16-channel PWM driver over I2C.
    """

    def __init__(self, address: int = 0x40, bus_num: int = 1,
                 simulate: bool = False):
        self.address = address
        self.simulate = simulate or not I2C_AVAILABLE
        self.bus = None

        if not self.simulate:
            self.bus = smbus2.SMBus(bus_num)
            self._init_device()
            log.info(f"PCA9685 initialized at 0x{address:02X} on bus {bus_num}")
        else:
            log.info(f"PCA9685 in SIMULATION mode (addr 0x{address:02X})")

    def _init_device(self):
        """Initialize PCA9685 — reset, set frequency for servos."""
        # Reset
        self._write_reg(PCA9685_REG.MODE1, PCA9685_REG.ALLCALL)
        self._write_reg(PCA9685_REG.MODE2, 0x04)  # Totem-pole outputs
        time.sleep(0.005)

        # Set PWM frequency to 50Hz (standard servo frequency)
        self.set_frequency(50)

    def _write_reg(self, reg: int, value: int):
        if not self.simulate:
            self.bus.write_byte_data(self.address, reg, value)

    def _read_reg(self, reg: int) -> int:
        if self.simulate:
            return 0
        return self.bus.read_byte_data(self.address, reg)

    def set_frequency(self, freq_hz: int):
        """
        Set PWM frequency. For servos, use 50Hz.
        Prescale = round(25MHz / (4096 * freq)) - 1
        """
        prescale = round(25_000_000.0 / (4096.0 * freq_hz)) - 1
        prescale = max(3, min(255, prescale))

        if not self.simulate:
            old_mode = self._read_reg(PCA9685_REG.MODE1)
            # Must sleep before changing prescale
            self._write_reg(PCA9685_REG.MODE1,
                           (old_mode & 0x7F) | PCA9685_REG.SLEEP)
            self._write_reg(PCA9685_REG.PRESCALE, prescale)
            self._write_reg(PCA9685_REG.MODE1, old_mode)
            time.sleep(0.005)
            # Restart
            self._write_reg(PCA9685_REG.MODE1,
                           old_mode | PCA9685_REG.RESTART | PCA9685_REG.AI)

        log.debug(f"PCA9685 frequency set to {freq_hz}Hz (prescale={prescale})")

    def set_pwm(self, channel: int, on: int, off: int):
        """
        Set PWM for a channel.
        Args:
            channel: 0-15
            on: 0-4095 (when in the cycle to turn on)
            off: 0-4095 (when in the cycle to turn off)
        """
        if channel < 0 or channel > 15:
            log.warning(f"Invalid PWM channel: {channel}")
            return

        reg_base = PCA9685_REG.LED0_ON_L + 4 * channel
        if not self.simulate:
            self.bus.write_byte_data(self.address, reg_base, on & 0xFF)
            self.bus.write_byte_data(self.address, reg_base + 1, on >> 8)
            self.bus.write_byte_data(self.address, reg_base + 2, off & 0xFF)
            self.bus.write_byte_data(self.address, reg_base + 3, off >> 8)

    def set_pwm_duty(self, channel: int, duty: float):
        """
        Set PWM duty cycle (0.0 - 1.0) for a channel.
        """
        off = int(duty * 4095)
        off = max(0, min(4095, off))
        self.set_pwm(channel, 0, off)

    def all_off(self):
        """Turn off all channels."""
        if not self.simulate:
            self._write_reg(PCA9685_REG.ALL_LED_ON_L, 0)
            self._write_reg(PCA9685_REG.ALL_LED_ON_H, 0)
            self._write_reg(PCA9685_REG.ALL_LED_OFF_L, 0)
            self._write_reg(PCA9685_REG.ALL_LED_OFF_H, 0)

    def cleanup(self):
        self.all_off()
        if self.bus:
            self.bus.close()

# ---------------------------------------------------------------------------
# AT24C02 EEPROM Driver
# ---------------------------------------------------------------------------
class AT24C02:
    """
    AT24C02 I2C EEPROM (256 bytes) for storing servo default positions.
    """

    def __init__(self, address: int = 0x50, bus_num: int = 1,
                 simulate: bool = False):
        self.address = address
        self.simulate = simulate or not I2C_AVAILABLE
        self.bus = None
        self._sim_data = bytearray(256)  # Simulated EEPROM

        if not self.simulate:
            self.bus = smbus2.SMBus(bus_num)
            log.info(f"AT24C02 EEPROM at 0x{address:02X}")
        else:
            log.info(f"AT24C02 in SIMULATION mode")

    def write_byte(self, mem_addr: int, value: int):
        """Write a single byte to EEPROM."""
        if self.simulate:
            self._sim_data[mem_addr] = value
            return
        self.bus.write_byte_data(self.address, mem_addr, value)
        time.sleep(0.005)  # EEPROM write cycle time

    def read_byte(self, mem_addr: int) -> int:
        """Read a single byte from EEPROM."""
        if self.simulate:
            return self._sim_data[mem_addr]
        return self.bus.read_byte_data(self.address, mem_addr)

    def write_block(self, start_addr: int, data: bytes):
        """Write a block of bytes (respecting page boundaries)."""
        for i, b in enumerate(data):
            self.write_byte(start_addr + i, b)

    def read_block(self, start_addr: int, length: int) -> bytes:
        """Read a block of bytes."""
        return bytes(self.read_byte(start_addr + i) for i in range(length))

    def cleanup(self):
        if self.bus:
            self.bus.close()

# ---------------------------------------------------------------------------
# Servo Controller (XR_Servo compatible API)
# ---------------------------------------------------------------------------

# Servo pulse width range at 50Hz (20ms period)
# Standard servo: 0.5ms (0°) to 2.5ms (180°)
# At 50Hz, 4096 ticks per period:
#   0.5ms  = 0.5/20 * 4096 ≈ 102
#   2.5ms  = 2.5/20 * 4096 ≈ 512
SERVO_MIN_PULSE = 102   # 0 degrees
SERVO_MAX_PULSE = 512   # 180 degrees

# EEPROM layout for saved angles:
# Byte 0: magic marker (0xAA = valid data)
# Bytes 1-8: servo angles for channels 1-8
EEPROM_MAGIC = 0xAA
EEPROM_MAGIC_ADDR = 0x00
EEPROM_ANGLES_START = 0x01

# Number of servo channels on this board
NUM_SERVOS = 8

# Default center position
DEFAULT_ANGLE = 90


class XR_Servo:
    """
    Drop-in replacement for _XiaoRGEEK_SERVO_.XR_Servo

    Drives up to 8 servos through the PCA9685 PWM generator
    on the XiaoR Geek PWR.A53 driver board.
    """

    def __init__(self, pca_address: int = 0x40, eeprom_address: int = 0x50,
                 bus_num: int = 1, simulate: bool = False):
        self.simulate = simulate
        self.pwm = PCA9685(address=pca_address, bus_num=bus_num,
                           simulate=simulate)
        self.eeprom = AT24C02(address=eeprom_address, bus_num=bus_num,
                              simulate=simulate)

        # Current angles for all 8 servos (1-indexed in API, 0-indexed internal)
        self.angles: List[int] = [DEFAULT_ANGLE] * NUM_SERVOS

        # Try to load saved angles from EEPROM
        self._load_saved_angles()

        log.info(f"XR_Servo initialized ({NUM_SERVOS} channels)")

    def _angle_to_pulse(self, angle: int) -> int:
        """Convert angle (0-180) to PCA9685 pulse count."""
        angle = max(0, min(180, angle))
        pulse = SERVO_MIN_PULSE + (angle / 180.0) * (SERVO_MAX_PULSE - SERVO_MIN_PULSE)
        return int(pulse)

    def _load_saved_angles(self):
        """Load saved servo angles from EEPROM if available."""
        try:
            magic = self.eeprom.read_byte(EEPROM_MAGIC_ADDR)
            if magic == EEPROM_MAGIC:
                for i in range(NUM_SERVOS):
                    angle = self.eeprom.read_byte(EEPROM_ANGLES_START + i)
                    if 0 <= angle <= 180:
                        self.angles[i] = angle
                log.info(f"Loaded saved angles from EEPROM: {self.angles}")
                # Apply saved angles
                for i in range(NUM_SERVOS):
                    self._set_channel(i, self.angles[i])
            else:
                log.info("No saved angles in EEPROM, using defaults")
        except Exception as e:
            log.warning(f"Could not read EEPROM: {e}")

    def _set_channel(self, channel: int, angle: int):
        """Set a PCA9685 channel to a specific angle."""
        pulse = self._angle_to_pulse(angle)
        self.pwm.set_pwm(channel, 0, pulse)

    # ----- Original API (compatible with _XiaoRGEEK_SERVO_.so) -----

    def XiaoRGEEK_SetServoAngle(self, ServoNum: int, angle: int):
        """
        Set servo angle.
        Args:
            ServoNum: 1-8 (matches original firmware numbering)
            angle: 15-160 (original firmware range, we support 0-180)
        """
        if ServoNum < 1 or ServoNum > NUM_SERVOS:
            log.warning(f"Invalid ServoNum: {ServoNum}")
            return

        angle = max(0, min(180, angle))
        channel = ServoNum - 1  # Convert to 0-indexed
        self.angles[channel] = angle
        self._set_channel(channel, angle)
        log.debug(f"Servo {ServoNum} -> {angle}°")

    def XiaoRGEEK_SaveServo(self):
        """Save current servo angles to EEPROM as power-on defaults."""
        try:
            self.eeprom.write_byte(EEPROM_MAGIC_ADDR, EEPROM_MAGIC)
            for i in range(NUM_SERVOS):
                self.eeprom.write_byte(EEPROM_ANGLES_START + i, self.angles[i])
            log.info(f"Saved angles to EEPROM: {self.angles}")
        except Exception as e:
            log.error(f"Failed to save to EEPROM: {e}")

    def XiaoRGEEK_ReSetServo(self):
        """Restore all servos to saved EEPROM positions."""
        self._load_saved_angles()
        log.info(f"Reset servos to saved positions: {self.angles}")

    # ----- Extended API for SignalBot -----

    def set_angle(self, servo_num: int, angle: int):
        """Pythonic wrapper for SetServoAngle."""
        self.XiaoRGEEK_SetServoAngle(servo_num, angle)

    def get_angle(self, servo_num: int) -> int:
        """Get current angle for a servo."""
        if 1 <= servo_num <= NUM_SERVOS:
            return self.angles[servo_num - 1]
        return -1

    def get_all_angles(self) -> dict:
        """Get all servo angles as a dict."""
        return {i + 1: self.angles[i] for i in range(NUM_SERVOS)}

    def sweep(self, servo_num: int, start: int, end: int,
              step: int = 1, delay: float = 0.02):
        """
        Smoothly sweep a servo from start to end angle.
        Useful for camera pan/tilt scanning.
        """
        if start < end:
            angles = range(start, end + 1, step)
        else:
            angles = range(start, end - 1, -step)

        for angle in angles:
            self.set_angle(servo_num, angle)
            time.sleep(delay)

    def center_all(self):
        """Center all servos to 90°."""
        for i in range(1, NUM_SERVOS + 1):
            self.set_angle(i, 90)
        log.info("All servos centered")

    def detach_all(self):
        """Turn off PWM to all servos (let them free-spin)."""
        self.pwm.all_off()
        log.info("All servos detached")

    def cleanup(self):
        """Clean shutdown."""
        self.pwm.cleanup()
        self.eeprom.cleanup()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    simulate = "--simulate" in sys.argv or not I2C_AVAILABLE

    logging.basicConfig(level=logging.DEBUG)
    print(f"Servo test ({'SIMULATION' if simulate else 'HARDWARE'} mode)")
    print()

    servo = XR_Servo(simulate=simulate)

    print("Current angles:", servo.get_all_angles())
    print()

    # Test: sweep servo 1
    print("Setting servo 1 to 30°...")
    servo.set_angle(1, 30)
    time.sleep(1)

    print("Saving positions...")
    servo.XiaoRGEEK_SaveServo()
    time.sleep(0.5)

    print("Setting servo 1 to 150°...")
    servo.set_angle(1, 150)
    time.sleep(1)

    print("Restoring saved positions...")
    servo.XiaoRGEEK_ReSetServo()
    time.sleep(1)

    print("Sweeping servo 1: 30° -> 150°...")
    servo.sweep(1, 30, 150, step=2, delay=0.03)

    print()
    print("Final angles:", servo.get_all_angles())

    servo.cleanup()
    print("Done.")
