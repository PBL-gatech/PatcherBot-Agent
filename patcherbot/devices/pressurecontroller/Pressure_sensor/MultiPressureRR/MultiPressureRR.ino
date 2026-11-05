/*
 * Configurable multi-channel pressure sensor reader.
 *
 * Wiring:
 *   Defaults: channel 0 uses A0, channel 1 uses A1 (if present).
 *   The host can assign any logical channel ID below maxChannels to any analog
 *   pin exposed in analogPins, using C commands below. Configuring another ID
 *   activates it automatically; capacity follows this board's analog inputs.
 *   UNO R4 exposes A0-A5; a Mega exposes A0-A15. Pin names are resolved on the
 *   Arduino, not assumed to be contiguous digital pin numbers by the host.
 *   Keep sensor outputs within the board's ADC input voltage range.
 *   Channel IDs here identify sensors, not pressure-box output channels.
 *
 * Serial protocol (9600 baud):
 *   Request:  one ASCII 'R' byte; no newline is required.
 *   Response: S0:<average>,1:<average>E\r\n
 *   Example:  S0:517.23,1:512.81E\r\n
 *   Configure: C<channel>:A<pin-number>\n, e.g. C2:A5\n.
 *   Success: K2:A5\r\n; failure: F:FORMAT, F:RANGE, or F:LENGTH plus CR/LF.
 *   Invalid configurations do not alter the pin mapping. A successful command
 *   resets only that channel's averaging history. Mapping is held in RAM and
 *   returns to defaults on reset; JSON initialization reapplies requested pins.
 *   All active channels are sampled once per R request, sequentially, not
 *   simultaneously. Values are raw ADC counts printed to two decimal places.
 *   Other bytes outside a C command, including CR/LF, are ignored.
 *   There is no unsolicited output or startup banner.
 *
 * Compatibility: this replaces the old S<number>E reply. Existing Python
 * readers expecting one float must be updated before using this firmware.
 * One host reader should own this serial connection and distribute channel
 * values; two independent readers must not compete for the same replies.
 *
 * Filtering and calibration:
 *   Each active channel has its own 25-sample moving-average buffer, sum, index, and
 *   sample count. During startup, divide by the number of real samples rather
 *   than 25, avoiding a false ramp caused by averaging initialized zeros.
 *   Sampling remains request-driven: at 20 requests/s the full window spans
 *   roughly 1.25 s. Changing the host polling rate changes this time window.
 *   An initial ADC conversion is discarded on each channel to allow settling
 *   after switching the ADC multiplexer between sensor inputs.
 *   Calibration belongs on the host, separately for each sensor. The old M/Y
 *   calculation was unused (the old sketch also transmitted raw averages),
 *   so it has been removed to make the transmitted units unambiguous.
 */

#include <stdlib.h>
#include <ctype.h>

// Use the board's named constants rather than assuming A0+n is portable.
// Extend this table if supporting a board exposing more than 16 analog inputs.
#if NUM_ANALOG_INPUTS > 16
#error "Extend analogPins for this board's additional analog inputs"
#endif
const int analogPins[] = {
  A0,
#if NUM_ANALOG_INPUTS > 1
  A1,
#endif
#if NUM_ANALOG_INPUTS > 2
  A2,
#endif
#if NUM_ANALOG_INPUTS > 3
  A3,
#endif
#if NUM_ANALOG_INPUTS > 4
  A4,
#endif
#if NUM_ANALOG_INPUTS > 5
  A5,
#endif
#if NUM_ANALOG_INPUTS > 6
  A6,
#endif
#if NUM_ANALOG_INPUTS > 7
  A7,
#endif
#if NUM_ANALOG_INPUTS > 8
  A8,
#endif
#if NUM_ANALOG_INPUTS > 9
  A9,
#endif
#if NUM_ANALOG_INPUTS > 10
  A10,
#endif
#if NUM_ANALOG_INPUTS > 11
  A11,
#endif
#if NUM_ANALOG_INPUTS > 12
  A12,
#endif
#if NUM_ANALOG_INPUTS > 13
  A13,
#endif
#if NUM_ANALOG_INPUTS > 14
  A14,
#endif
#if NUM_ANALOG_INPUTS > 15
  A15,
#endif
};
const int maxChannels = sizeof(analogPins) / sizeof(analogPins[0]);
const int numReadings = 25;
int sensorPins[maxChannels];  // -1 means this logical channel is inactive.

// Static storage starts at zero. Integer samples and 32-bit sums avoid storing
// floats in the history and keep the sum safe on AVR's 16-bit int platform.
int readings[maxChannels][numReadings] = {};
unsigned long totals[maxChannels] = {};
int readIndices[maxChannels] = {};
int sampleCounts[maxChannels] = {};
char configBuffer[32];
int configLength = 0;
bool receivingConfig = false;
bool configOverflow = false;

void configureChannel() {
  if (configOverflow) {
    Serial.println("F:LENGTH");
    return;
  }
  configBuffer[configLength] = '\0';
  char *end;
  const unsigned long channel = strtoul(configBuffer, &end, 10);
  if (!isdigit(configBuffer[0]) || *end != ':' || end[1] != 'A') {
    Serial.println("F:FORMAT");
    return;
  }
  char *pinStart = end + 2;
  const unsigned long pinIndex = strtoul(pinStart, &end, 10);
  if (!isdigit(*pinStart) || *end != '\0') {
    Serial.println("F:FORMAT");
    return;
  }
  if (channel >= (unsigned long)maxChannels || pinIndex >= (unsigned long)maxChannels) {
    Serial.println("F:RANGE");
    return;
  }
  sensorPins[channel] = analogPins[pinIndex];
  totals[channel] = 0;
  readIndices[channel] = 0;
  sampleCounts[channel] = 0;
  for (int i = 0; i < numReadings; i++) {
    readings[channel][i] = 0;
  }
  Serial.print("K");
  Serial.print(channel);
  Serial.print(":A");
  Serial.println(pinIndex);
}

void setup() {
  Serial.begin(9600);
  for (int channel = 0; channel < maxChannels; channel++) {
    sensorPins[channel] = -1;
  }
  sensorPins[0] = analogPins[0];
  if (maxChannels > 1) {
    sensorPins[1] = analogPins[1];
  }
  // UNO R4 supports higher resolutions, but use explicit 10-bit counts (0-1023)
  // to retain the scale of the original reader. Changing this requires sensor
  // recalibration on the host. This sketch now targets the UNO R4 WiFi.
#if defined(ARDUINO_ARCH_RENESAS) || defined(ARDUINO_ARCH_RENESAS_UNO)
  analogReadResolution(10);
#endif
}

void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  const char command = Serial.read();
  if (receivingConfig) {
    if (command == '\n') {
      configureChannel();
      receivingConfig = false;
    } else if (command != '\r') {
      if (configLength < (int)sizeof(configBuffer) - 1) {
        configBuffer[configLength++] = command;
      } else {
        configOverflow = true;  // Consume through newline before rejecting.
      }
    }
    return;
  }
  if (command == 'C') {
    receivingConfig = true;
    configLength = 0;
    configOverflow = false;
    return;
  }
  if (command != 'R') {
    return;
  }

  // Acquire both readings before transmitting so serial output does not add
  // a delay between the channel samples. Each R advances both filters once.
  float averages[maxChannels];
  for (int channel = 0; channel < maxChannels; channel++) {
    if (sensorPins[channel] < 0) {
      continue;
    }
    analogRead(sensorPins[channel]);  // Discard conversion after pin switch.
    const int sensorValue = analogRead(sensorPins[channel]);
    const int index = readIndices[channel];

    totals[channel] -= readings[channel][index];
    readings[channel][index] = sensorValue;
    totals[channel] += sensorValue;
    readIndices[channel] = (index + 1) % numReadings;

    if (sampleCounts[channel] < numReadings) {
      sampleCounts[channel]++;
    }
    averages[channel] = (float)totals[channel] / sampleCounts[channel];
  }

  // Explicit IDs and framing let the host associate each value with its sensor.
  Serial.print("S");
  bool first = true;
  for (int channel = 0; channel < maxChannels; channel++) {
    if (sensorPins[channel] < 0) {
      continue;
    }
    if (!first) {
      Serial.print(",");
    }
    first = false;
    Serial.print(channel);
    Serial.print(":");
    Serial.print(averages[channel], 2);
  }
  Serial.println("E");
}
