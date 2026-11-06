// SSVEP stimulus: flickers 5 LEDs at exact, independent frequencies.
//
// Uses integer microsecond bookkeeping with a remainder, so the frequencies
// stay exact over long sessions instead of slowly drifting.
//
// Wiring: each LED pin -> 220-330 ohm resistor -> LED -> GND.
// Bigger, brighter targets give stronger brain responses, so diffuse each LED
// behind a few cm of frosted plastic or paper.
//
// Serial commands (115200 baud): '1' = flicker on, '0' = all LEDs off.
//
// SAFETY: flickering light in this range can trigger seizures in people with
// photosensitive epilepsy. Warn everyone before they look at it.

const int NUM_LEDS = 5;
const int LED_PINS[NUM_LEDS] = {3, 5, 6, 9, 10};
const unsigned long FREQ_HZ[NUM_LEDS] = {7, 9, 11, 13, 15};   // must match config.py

unsigned long nextToggle[NUM_LEDS];
unsigned long remainderAcc[NUM_LEDS];
bool ledState[NUM_LEDS];
bool running = true;

void scheduleNext(int i) {
  // Half period = 1,000,000 / (2 f) microseconds, kept exact with a remainder.
  unsigned long denom = 2 * FREQ_HZ[i];
  nextToggle[i] += 1000000UL / denom;
  remainderAcc[i] += 1000000UL % denom;
  if (remainderAcc[i] >= denom) {
    nextToggle[i] += 1;
    remainderAcc[i] -= denom;
  }
}

void resetTimers() {
  unsigned long now = micros();
  for (int i = 0; i < NUM_LEDS; i++) {
    nextToggle[i] = now;
    remainderAcc[i] = 0;
    ledState[i] = false;
    digitalWrite(LED_PINS[i], LOW);
    scheduleNext(i);
  }
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < NUM_LEDS; i++) pinMode(LED_PINS[i], OUTPUT);
  resetTimers();
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '0') {
      running = false;
      for (int i = 0; i < NUM_LEDS; i++) digitalWrite(LED_PINS[i], LOW);
    } else if (c == '1') {
      running = true;
      resetTimers();
    }
  }
  if (!running) return;

  unsigned long now = micros();
  for (int i = 0; i < NUM_LEDS; i++) {
    // Signed difference handles micros() rollover (~70 minutes).
    if ((long)(now - nextToggle[i]) >= 0) {
      ledState[i] = !ledState[i];
      digitalWrite(LED_PINS[i], ledState[i] ? HIGH : LOW);
      scheduleNext(i);
    }
  }
}
